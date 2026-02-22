import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image_folder = "ToBeAnalysed"

valid_extensions = (".jpg", ".jpeg", ".png")

files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
total_files = len(files)

print(f"Nalezeno {total_files} obrázků.\n")

data = []

for index, filename in enumerate(files, start=1):
    filepath = os.path.join(image_folder, filename)

    try:
        progress = (index / total_files) * 100
        print(f"[{index}/{total_files}] ({progress:.2f} %) Zpracovávám: {filename}")

        name_without_extension = os.path.splitext(filename)[0]
        species_name = name_without_extension.split("_")[0]

        img = plt.imread(filepath)

        if img.shape[-1] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            img = img.astype(np.float32)

        mean_r = np.mean(img[:, :, 0])
        mean_g = np.mean(img[:, :, 1])
        mean_b = np.mean(img[:, :, 2])

        luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        mean_brightness = np.mean(luminance)
        contrast = np.std(luminance)

        data.append([
            species_name,
            filename,
            mean_r,
            mean_g,
            mean_b,
            mean_brightness,
            contrast
        ])

    except Exception as e:
        print(f"Chyba při zpracování {filename}: {e}")

print("\nVytvářím DataFrame...")

df = pd.DataFrame(data, columns=[
    "species",
    "filename",
    "mean_R",
    "mean_G",
    "mean_B",
    "mean_brightness",
    "contrast"
])

df.to_csv("butterfly_features.csv", index=False)

print("\nHotovo. Dataset vytvořen.")
print(df.head())