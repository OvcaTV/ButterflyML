import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Konfigurace

MODEL_PATH = "motyliModel.keras"
IMG_SIZE   = (384, 384)
CONFIDENCE_THRESHOLD  = 0.30


CLASS_NAMES = ['BabockaAdmiral',
               'BabockaBileC',
               'BabockaBodlakov',
               'BabockaBodlakova',
               'BabockaKoprivova',
               'BabockaPaviOko',
               'BabockaSitkovana',
               'BatolecCerveny',
               'BatolecDuhovy',
               'BekyneVelkohlava',
               'BekyneZlatoritna',
               'Belasek',
               'BelasekHrachorovy',
               'BelasekOvocny',
               'BelasekRepkovy',
               'BelasekRepovy',
               'BelasekRerichovy',
               'BelasekRezedkovy',
               'BelasekZelny',
               'BeloskvrnacLisejnkovy',
               'BeloskvrnacPampelskovy',
               'HnedasekJitroceloy',
               'HnedasekRozraziloy',
               'LisejnikovecVroubny',
               'ModrasekCernolemy',
               'ModrasekJehlicovy',
               'ModrasekKrusinovy',
               'ModrasekPodobny',
               'ModrasekStirovnikvy',
               'ModrasekTmavohned',
               'ModrasekUslechtil',
               'ModrasekVikvicovy',
               'OhnivacekCelikovy',
               'OhnivacekCernocary',
               'OhnivacekCernokrily',
               'OhnivacekCernoskvnny',
               'OhnivacekModrolem',
               'OhnivacekModrolesly',
               'Okac',
               'OkacBojinkovy',
               'OkacCernohnedy',
               'OkacJecminkovy',
               'OkacLucni',
               'OkacOvsovy',
               'OkacPohankovy',
               'OkacProsickovy',
               'OkacPyrovy',
               'OkacRudopasny',
               'OkacTreslicovy',
               'OkacZedni',
               'OstruhacekJilmovy',
               'OtakarekFenyklovy',
               'OtakarekOvocny',
               'OtakarekOvocnyMlySedlec',
               'Papilio Memnon',
               'PapilioDemoleus',
               'PapilioLowi',
               'PapilioMemnon',
               'PapilioPalinurus',
               'PapilioPolytes',
               'Perletovec',
               'PerletovecFialkov',
               'PerletovecKoprivoy',
               'PerletovecMaly',
               'PerletovecNejmens',
               'PerletovecOstruziovy',
               'PerletovecProstreni',
               'PerletovecStribroasek',
               'PerletovecStribroasekValestina',
               'PerletovecVelky',
               'PrastevnikChrastacovy',
               'PrastevnikHluchavovy',
               'PrastevnikJitroceovy',
               'PrastevnikKostivaovy',
               'PrastevnikStarckoy',
               'Soumracik',
               'Soumracnik',
               'SoumracnikCareckoany',
               'SoumracnikCernohndy',
               'SoumracnikJitroceovy',
               'SoumracnikMakovy',
               'SoumracnikMetlicoy',
               'SoumracnikRezavy',
               'SoumracnikSlezovy)',
               'VretenukaObecna',
               'Vretenuska',
               'VretenuskaCicorkoa',
               'VretenuskaKozincoa',
               'VretenuskaLigrusoa',
               'VretenuskaMateriduskova',
               'VretenuskaObecna',
               'VretenuskaPetiteca',
               'VretenuskaPozdni',
               'VretenuskaPrehlizna',
               'Zelenacek',
               'ZelenacekStovikov',
               'ZlutasekBoruvkovy)',
               'ZlutasekCicoreckoy',
               'ZlutasekCilimnikoy',
               'ZlutasekResetlakoy'
               ]


@tf.keras.utils.register_keras_serializable(package="butterfly")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr=1e-7, name=None):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
            lr = self.min_lr + (self.peak_lr - self.min_lr) * scale
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
            lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        return lr

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


#nacteni modelu
print(f"Nacitam model: {MODEL_PATH} ...")
_model = load_model(MODEL_PATH)
print("Model nacten.\n")


def _load_image(image_path: str) -> tf.Tensor:
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32)  # raw [0, 255] — backbone normalises internally


def _tta_augment(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.08 * 255)
    img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
    return tf.clip_by_value(img, 0.0, 255.0)


def predict_butterfly(image_path: str,n_tta: int = 8,top_k: int = 3,confidence_threshold: float = CONFIDENCE_THRESHOLD,) -> None:
    """
    Parametry
    image_path: umisteni obrazku, duh (JPEG or PNG)
    n_tta : number of forward passes to average (1 = no TTA, faster)
    top_k : kolik nejlepsich vysledku vypsat
    confidence_threshold : jak moc si musi byt model minimalne jisty druhem motyla
    """

    if not CLASS_NAMES:
        raise ValueError(
            "CLASS_NAMES is empty. Open this file and paste your species list "
            "from the training output into the CLASS_NAMES list at the top."
        )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Obrazek nenalezen: '{image_path}, vyzkousej jine umisteni'\n")

    img = _load_image(image_path)

    # Pass 1: clean image -- in-model augmentation is off (training=False)
    passes = [_model(img[tf.newaxis], training=False).numpy()]

    # Passes 2-n: lightly augmented versions
    for _ in range(n_tta - 1):
        passes.append(_model(_tta_augment(img)[tf.newaxis], training=False).numpy())

    avg_probs = np.mean(passes, axis=0)[0]
    top_idx = np.argsort(avg_probs)[::-1][:top_k]
    best_conf = avg_probs[top_idx[0]]

    print(f"Odhad pro: {image_path}")
    print("-" * 55)


    if best_conf < confidence_threshold:
        print("Na obrazku nebyl detekovan motyl")
        return

    for rank, idx in enumerate(top_idx, 1):
        confidence = avg_probs[idx] * 100
        bar = "|" * int(confidence / 5)
        print(f"  {rank}. {CLASS_NAMES[idx]:<35} {confidence:5.1f}%  {bar}")
    print()



# Pouze tohle se mění

if __name__ == "__main__":
    predict_butterfly("TestFotky/IMG_20260314_112917.jpg")
