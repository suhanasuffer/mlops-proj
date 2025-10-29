import os, shutil
from pathlib import Path
import yaml

params = yaml.safe_load(open("params.yaml"))
processed_dir = Path(params["data"]["processed_dir"])
raw_dir = Path(params["data"]["raw_dir"])
processed_dir.mkdir(parents=True, exist_ok=True)

print(f"Copying raw data from {raw_dir} to {processed_dir} ...")
if processed_dir.exists():
    shutil.rmtree(processed_dir)
shutil.copytree(raw_dir, processed_dir)
print("Preprocessing complete.")
