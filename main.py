import os
import numpy as np
import pandas as pd
from PIL import Image

# Folder with butterfly images
IMAGE_FOLDER = "ToBeAnalysed"

# Output dataset file
OUTPUT_FILE = "dataset.csv"


def extract_filename_info(filename):
    """
    Extract species and position from filename:
    Example:
    Monarch_random_top.jpg

    Species = Monarch
    Position = top
    """

    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    species = parts[0] if len(parts) > 0 else "unknown"
    position = parts[1] if len(parts) > 1 else "unknown"

    return species, position


def analyze_image(filepath):

    img = Image.open(filepath).convert("RGB")
    img_array = np.array(img)

    # Flatten pixels
    pixels = img_array.reshape(-1, 3)

    # Average color
    avg_color = pixels.mean(axis=0)
    avg_r, avg_g, avg_b = avg_color

    # Dominant color (most common pixel)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]
    dom_r, dom_g, dom_b = dominant

    # Color contrast (standard deviation)
    contrast = pixels.std()

    # Color histogram (16 bins per channel)
    hist_r, _ = np.histogram(pixels[:,0], bins=16, range=(0,255))
    hist_g, _ = np.histogram(pixels[:,1], bins=16, range=(0,255))
    hist_b, _ = np.histogram(pixels[:,2], bins=16, range=(0,255))

    histogram = np.concatenate([hist_r, hist_g, hist_b])

    return {
        "avg_r": avg_r,
        "avg_g": avg_g,
        "avg_b": avg_b,
        "dom_r": dom_r,
        "dom_g": dom_g,
        "dom_b": dom_b,
        "contrast": contrast,
        **{f"hist_{i}": histogram[i] for i in range(len(histogram))}
    }


data = []
count = 0

all_files = [f for f in os.listdir(IMAGE_FOLDER)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]

total_photos = len(all_files)

print(f"Found {total_photos} images to analyze.\n")

for filename in all_files:

    filepath = os.path.join(IMAGE_FOLDER, filename)

    species, position = extract_filename_info(filename)
    features = analyze_image(filepath)

    row = {
        "image_name": filename,
        "species": species,
        "position": position,
        **features
    }

    data.append(row)

    count += 1
    print(f"Processed {count}/{total_photos}: {filename}")

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)

print("\n---------------------------------")
print(f"Finished! {count} photos analyzed.")
print(f"Dataset saved to: {OUTPUT_FILE}")