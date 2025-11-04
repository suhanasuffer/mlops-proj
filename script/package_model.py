# script/package_model.py
import os
import yaml
import shutil
import json
import tarfile

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    params = load_params()
    model_path = params["training"]["model_output"]
    packaged_model_dir = "models"
    packaged_model_path = os.path.join(packaged_model_dir, "packaged_model.h5")

    os.makedirs(packaged_model_dir, exist_ok=True)

    # Copy trained model to packaged model location
    if os.path.exists(model_path):
        shutil.copy(model_path, packaged_model_path)
        print(f"📦 Model packaged successfully at: {packaged_model_path}")
    else:
        raise FileNotFoundError(f"❌ Trained model not found at {model_path}")

    # Save metadata about the model packaging
    metadata = {
        "model_source": model_path,
        "packaged_path": packaged_model_path,
        "status": "success"
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(" Model metadata saved at reports/model_metadata.json")

    # Create deployable tar.gz archive for DVC stage
    os.makedirs("deploy", exist_ok=True)
    with tarfile.open("deploy/model_package.tar.gz", "w:gz") as tar:
        tar.add(packaged_model_path, arcname="packaged_model.h5")

    print("Packaged model archive created at: deploy/model_package.tar.gz")
