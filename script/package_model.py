import tarfile, os

os.makedirs("deploy", exist_ok=True)
with tarfile.open("deploy/model_package.tar.gz", "w:gz") as tar:
    tar.add("models/solar_model.h5", arcname="solar_model.h5")
print("Model packaged successfully.")
