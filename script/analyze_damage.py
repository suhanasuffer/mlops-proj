# script/analyze_damage.py
import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def reconstruction_error(original, reconstructed):
    """Compute per-image mean squared reconstruction error."""
    return np.mean(np.square(original - reconstructed), axis=(1, 2, 3))

if __name__ == "__main__":
    params = load_params()
    img_height = params["training"]["img_height"]
    img_width = params["training"]["img_width"]
    batch_size = params["training"]["batch_size"]
    model_path = params["training"]["model_output"]

    print("📂 Loading processed validation images...")
    processed_dir = "data/processed"

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    test_gen = datagen.flow_from_directory(
        processed_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("🧠 Loading trained autoencoder model safely...")
    # explicitly define the mse function for deserialization
    autoencoder = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    print("🔍 Computing reconstruction errors...")
    all_errors = []
    y_true = test_gen.classes

    test_gen.reset()
    for i in tqdm(range(len(test_gen))):
        batch = test_gen[i][0]
        reconstructed = autoencoder.predict(batch)
        errors = reconstruction_error(batch, reconstructed)
        all_errors.extend(errors)

    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    threshold = mean_error + 2 * std_error

    print(f"📏 Damage threshold (mean + 2*std): {threshold:.4f}")

    y_pred = (all_errors > threshold).astype(int)
    damage_labels = np.array(['Healthy' if e <= threshold else 'Damaged' for e in all_errors])

    os.makedirs("reports", exist_ok=True)

    # Histogram of reconstruction errors
    plt.figure(figsize=(8, 5))
    plt.hist(all_errors, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.title('Distribution of Reconstruction Errors')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/reconstruction_error_hist.png")

    # Confusion matrix (optional visualization)
    cm = confusion_matrix(y_true, y_pred > 0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Autoencoder Threshold)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix_autoencoder.png")

    # Save aggregate results
    results = {
        "mean_error": float(mean_error),
        "std_error": float(std_error),
        "threshold": float(threshold)
    }
    with open("reports/autoencoder_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    # ✅ Save individual reconstruction errors and labels
    errors_file = "reports/reconstruction_errors.json"
    class_indices = {v: k for k, v in test_gen.class_indices.items()}
    label_names = [class_indices[c] for c in y_true]

    with open(errors_file, "w") as f:
        json.dump({
            "errors": all_errors.tolist(),
            "labels": label_names
        }, f, indent=2)

    print(f" Saved individual reconstruction errors to {errors_file}")
    print(" Analysis complete. Results saved in 'reports/' folder.")
