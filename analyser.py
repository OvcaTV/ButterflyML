"""
Vzhled slozky s fotografiema
  <DATA_DIR>/
      SpeciesA_photo001.jpg
      SpeciesA_photo002.jpg
      SpeciesB_photo001.jpg
      atd...
"""

import math
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.applications import EfficientNetV2S
from keras.callbacks import EarlyStopping
from keras.layers import (
    BatchNormalization, Dense, Dropout,
    GlobalAveragePooling2D, RandomContrast, RandomFlip,
    RandomRotation, RandomZoom,
)
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# 1.  CONFIGURATION

DATA_DIR = "ToBeAnalysed"   # folder with all butterfly photos
IMG_SIZE = (384, 384)      # EfficientNetV2S native resolution
BATCH_SIZE = 128              # needs to be large enough to cover many of the 100
                             # classes per batch — 64 minimum, 128 if VRAM allows
SEED = 1
MODEL_PATH = "butterfly_model_v2.keras"

VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

# Learning rate schedule
# P1_LR → 1e-3: with 100 classes and noisy gradients,
P1_LR = 1e-3   # phase 1 peak LR  (head only, backbone frozen)
P2_LR = 3e-5   # phase 2 peak LR  (fine-tuning top layers)
P1_MAX_EPOCHS = 100     # more epochs to compensate for slower, steadier LR
P2_MAX_EPOCHS = 80
WARMUP_EPOCHS = 5      # longer warmup smooths out early instability
ES_PATIENCE = 12     # slightly more patience since convergence is slower

LABEL_SMOOTHING = 0.05
N_FINE_TUNE     = 40    # how many top backbone layers to unfreeze in phase 2

AUTOTUNE = tf.data.AUTOTUNE


# 2.  MIXED PRECISION

# Tensors are computed in float16, weights stored in float32.
# Requires an Nvidia GPU with compute capability ≥ 7.0 (Volta / Turing /
# Ampere / Ada).  On CPU or older GPUs, comment this line out — it will
# still run correctly but give no speedup.

tf.keras.mixed_precision.set_global_policy("mixed_float16")
print(f"Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}\n")


# 3.  DATA LOADING

all_files = [
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
all_paths = [os.path.join(DATA_DIR, f) for f in all_files]
all_labels = [f.split('_')[0] for f in all_files]

counts = Counter(all_labels)
dropped = [s for s, n in counts.items() if n < 2]
if dropped:
    print(f"Dropped {len(dropped)} species with < 2 images: {dropped}")

filtered = [(p, l) for p, l in zip(all_paths, all_labels) if counts[l] >= 2]
all_paths, all_labels = zip(*filtered)

le = LabelEncoder()
all_labels_enc = le.fit_transform(all_labels)
CLASS_NAMES = list(le.classes_)
NUM_CLASSES = len(CLASS_NAMES)

print(f"Detected {NUM_CLASSES} species: {CLASS_NAMES}\n")


# 4.  TRAIN / VAL / TEST SPLIT

X_temp, X_test, y_temp, y_test = train_test_split(
    all_paths, all_labels_enc,
    test_size=TEST_SPLIT, stratify=all_labels_enc, random_state=SEED,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SPLIT, stratify=y_temp, random_state=SEED,
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")


# 5.  CLASS WEIGHTS
# "balanced" gives weight ∝ 1 / class_frequency, so rare species contribute
# the same total gradient as common ones.

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)
class_weight_dict = dict(enumerate(class_weights_arr))
print("Class weights:", {CLASS_NAMES[k]: f"{v:.2f}" for k, v in class_weight_dict.items()}, "\n")

# 6.  tf.data PIPELINE
# Augmentation lives inside the model (see section 8), so the pipeline only
# handles loading, resizing, and normalisation.

def preprocess(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    # EfficientNetV2S includes its own preprocessing layer internally
    # (include_preprocessing=True by default).  Pass raw [0, 255] float values
    # and let the backbone rescale — dividing by 255 here would double-normalise
    # and collapse all activations to near-zero.
    img = tf.cast(img, tf.float32)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels.tolist()))
    if training:
        ds = ds.shuffle(buffer_size=2_000, seed=SEED)
    ds = (ds
          .map(preprocess, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))
    return ds


train_ds = make_dataset(X_train, y_train, training=True)
val_ds = make_dataset(X_val,   y_val)
test_ds = make_dataset(X_test,  y_test)


# 7.  LR SCHEDULE: COSINE DECAY WITH LINEAR WARMUP
# @register_keras_serializable makes the class discoverable when loading a
# saved model without manually passing custom_objects every time.

@tf.keras.utils.register_keras_serializable(package="butterfly")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        # Linear ramp from 0 → peak_lr over warmup_steps
        warmup   = self.peak_lr * (step / float(self.warmup_steps))
        # Cosine decay from peak_lr → min_lr over the remaining steps
        progress = (step - self.warmup_steps) / float(
            max(self.total_steps - self.warmup_steps, 1)
        )
        cosine   = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * tf.clip_by_value(progress, 0.0, 1.0))
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

p1_schedule = WarmupCosineDecay(
    peak_lr=P1_LR,
    total_steps=steps_per_epoch * P1_MAX_EPOCHS,
    warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
)
p2_schedule = WarmupCosineDecay(
    peak_lr=P2_LR,
    total_steps=steps_per_epoch * P2_MAX_EPOCHS,
    warmup_steps=steps_per_epoch * WARMUP_EPOCHS,
)


# 8.  MODEL

# Architecture:
#   Input → [Augmentation block] → EfficientNetV2S → GAP → BN → Dense(512)
#        → Dropout → Dense(256) → Dropout → Softmax
#
# The augmentation block uses standard Keras preprocessing layers.  Keras
# knows how to serialise these correctly, avoiding the EagerTensor JSON error.
# More importantly, Keras automatically disables them when the model is called
# with training=False — no manual flag management needed.

def build_model(num_classes: int, img_size: tuple) -> tuple[Model, Model]:
    inputs = tf.keras.Input(shape=(*img_size, 3), name="image_input")

    # Augmentation inside the model
    x = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.12),
        RandomZoom(0.10),
        RandomContrast(0.10),
    ], name="augmentation")(inputs)

    # EfficientNetV2S backbone (called as a layer so its input is `x`)
    base = EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base.trainable = False   # frozen for phase 1
    x = base(x)

    # Classification head
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="bn")(x)
    x = Dense(512, activation="relu", name="dense_512")(x)
    x = Dropout(0.4, name="drop_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = Dropout(0.3, name="drop_2")(x)

    # float32 output keeps softmax numerically stable under mixed precision
    outputs = Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="butterfly_classifier")
    return model, base


model, base_model = build_model(NUM_CLASSES, IMG_SIZE)
model.summary(show_trainable=True)


# 9.  PHASE 1 — TRAIN HEAD (frozen backbone)

model.compile(
    optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(p1_schedule)
    ),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),   # ⑥
    metrics=["accuracy"],
)

print("\n=== Phase 1: Training classification head ===\n")

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=P1_MAX_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(
            monitor="val_accuracy",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ],
)

# 10.  PHASE 2 — FINE-TUNE (unfreeze top N backbone layers)

print(f"\n=== Phase 2: Fine-tuning top {N_FINE_TUNE} backbone layers ===\n")

base_model.trainable = True
for layer in base_model.layers[:-N_FINE_TUNE]:
    layer.trainable = False

# Recompile with a much lower LR to avoid overwriting pre-trained weights
model.compile(
    optimizer=tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(p2_schedule)
    ),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=P2_MAX_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(
            monitor="val_accuracy",
            patience=ES_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )
    ],
)

#ulozeni modelu
model.save(MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}\n")