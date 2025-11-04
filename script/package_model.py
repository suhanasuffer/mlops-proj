# script/package_model.py
import tarfile, os, yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)
model_path = params["training"]["model_output"]
os.makedirs("deploy", exist_ok=True)
out = "deploy/model_package.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    tar.add(model_path, arcname=os.path.basename(model_path))
print("Packaged model:", out)
