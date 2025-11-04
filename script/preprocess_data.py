# script/preprocess_data.py
import os
import cv2
import yaml
import numpy as np
from tqdm import tqdm

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_images(input_root, output_root, img_height, img_width):
    classes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    print(f" Found classes: {classes}")

    for cls in classes:
        in_dir = os.path.join(input_root, cls)
        out_dir = os.path.join(output_root, cls)
        ensure_dir(out_dir)

        for fname in tqdm(os.listdir(in_dir), desc=f"Processing {cls}"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(in_dir, fname)
                try:
                    img = cv2.imread(fpath)
                    if img is None:
                        print(f"⚠ Could not read {fpath}")
                        continue
                    img = cv2.resize(img, (img_width, img_height))
                    out_path = os.path.join(out_dir, fname)
                    cv2.imwrite(out_path, img)
                except Exception as e:
                    print("❌ Error:", e)

if __name__ == "__main__":
    params = load_params()
    raw_dir = "data/raw/Faulty_solar_panel"
    processed_dir = "data/processed"

    img_height = params["training"]["img_height"]
    img_width = params["training"]["img_width"]

    ensure_dir(processed_dir)
    preprocess_images(raw_dir, processed_dir, img_height, img_width)
    print(" Preprocessing completed.")
