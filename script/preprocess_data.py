# script/preprocess_data.py
import os, cv2, yaml, numpy as np
from pathlib import Path

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def process_and_save(src_root, dst_root, params):
    ensure_dir(dst_root)
    h = params["training"]["img_height"]
    w = params["training"]["img_width"]
    augment = params["data"].get("data_augmentation", False)
    for root, dirs, files in os.walk(src_root):
        # detect class folder 
        rel = os.path.relpath(root, src_root)
        out_dir = os.path.join(dst_root, rel) if rel != "." else dst_root
        ensure_dir(out_dir)
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                src_path = os.path.join(root, fname)
                try:
                    img = cv2.imread(src_path)
                    if img is None:
                        print(f"Warning: cannot read {src_path}")
                        continue
                    img = cv2.resize(img, (w, h))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = (img / 255.0).astype("float32")
                    # save as .npy or jpg scaled back to 0-255; here save as jpg for simplicity
                    out_path = os.path.join(out_dir, fname)
                    img_to_save = (img * 255).astype("uint8")
                    cv2.imwrite(out_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    # optional augmentation
                    if augment:
                        # horizontal flip
                        flip = cv2.flip(img_to_save, 1)
                        fn, ext = os.path.splitext(fname)
                        cv2.imwrite(os.path.join(out_dir, f"{fn}_flip{ext}"),
                                    cv2.cvtColor(flip, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    print("Error processing", src_path, e)

if __name__ == "__main__":
    params = load_params()
    raw_dir = params["data"].get("raw_dir", "data/raw")
    processed_dir = params["data"].get("processed_dir", "data/processed")
    # If raw_dir contains class subfolders, it will preserve them.
    process_and_save(raw_dir, processed_dir, params)
    print("Preprocessing done.")
