import yaml, time

params = yaml.safe_load(open("params.yaml"))
epochs = params["training"]["epochs"]
lr = params["training"]["learning_rate"]
model_path = params["training"]["model_output"]

print(f"Training model for {epochs} epochs at lr={lr}")
time.sleep(2)
open(model_path, "w").write("dummy model data")
print(f"Saved model to {model_path}")
