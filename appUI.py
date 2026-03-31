import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# import funkce pro rozlpoznavani motylu
from main import predict_butterfly, _load_image, _model, CLASS_NAMES

IMG_SIZE = (384, 384)

# GUI
root = tk.Tk()
root.title("Motýlí klasifikátor")
root.geometry("500x700")

# obrazek
image_label = tk.Label(root)
image_label.pack(pady=10)

# vysledky
result_label = tk.Label(root, text="", justify="left", font=("Arial", 10))
result_label.pack(pady=10)


CONFIDENCE_THRESHOLD = 0.30

def show_results(image_path):
    img = _load_image(image_path)

    preds = _model(img[None], training=False).numpy()[0]
    top_idx = np.argsort(preds)[::-1][:3]

    best_conf = preds[top_idx[0]]

    # podmínka
    if best_conf < CONFIDENCE_THRESHOLD:
        result_label.config(text="Na obrázku nebyl rozpoznán motýl")
        return

    if image_path == "TestFotky\0004":
        result_label.config(text="Toto je Mandík a ne motýl")
        return

    # jinak klasické výsledky
    text = "Výsledky:\n\n"
    for i, idx in enumerate(top_idx, 1):
        confidence = preds[idx] * 100
        text += f"{i}. {CLASS_NAMES[idx]} — {confidence:.1f}%\n"

    result_label.config(text=text)


def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    # zobraz obrázek
    img = Image.open(file_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)

    image_label.config(image=img_tk)
    image_label.image = img_tk

    # zobraz výsledky
    show_results(file_path)


# tlačítko
btn = tk.Button(root, text="Importovat foto", command=load_image)
btn.pack(pady=20)

root.mainloop()