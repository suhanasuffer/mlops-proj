# script/evaluate_threshold.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def evaluate_cleanliness_threshold(errors_file="reports/autoencoder_evaluation.json",
                                   scores_file="reports/reconstruction_errors.json",
                                   save_dir="results",
                                   alpha=0.05):
    """Evaluates whether reconstruction error threshold separates clean vs dusty panels."""
    os.makedirs(f"{save_dir}/metrics", exist_ok=True)
    os.makedirs(f"{save_dir}/figures", exist_ok=True)

    # Load saved threshold and errors
    with open(errors_file, "r") as f:
        eval_data = json.load(f)
    threshold = eval_data["threshold"]

    with open(scores_file, "r") as f:
        scores_data = json.load(f)
    all_errors = np.array(scores_data["errors"])
    labels = np.array(scores_data["labels"])

    # Split into clean vs dusty groups
    clean_errors = all_errors[np.array(labels) == "Clean"]
    dusty_errors = all_errors[np.array(labels) != "Clean"]

    if len(clean_errors) < 2 or len(dusty_errors) < 2:
        print("❌ Not enough data points for t-test.")
        return

    # Welch’s t-test
    t_stat, p_value = stats.ttest_ind(clean_errors, dusty_errors, equal_var=False)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

    interpretation = (
        "✅ Significant difference — threshold distinguishes Clean vs Dusty panels."
        if p_value < alpha
        else "⚠ No significant difference — adjust threshold."
    )
    print(interpretation)

    # Save numeric results
    with open(f"{save_dir}/metrics/threshold_eval.json", "w") as f:
        json.dump({
            "t_stat": t_stat,
            "p_value": p_value,
            "alpha": alpha,
            "threshold": threshold
        }, f, indent=2)

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.hist(clean_errors, bins=20, alpha=0.7, label='Clean', color='green')
    plt.hist(dusty_errors, bins=20, alpha=0.7, label='Dusty/Dirty', color='brown')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution: Clean vs Dusty Panels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/reconstruction_hist.png")

    plt.figure(figsize=(6, 5))
    plt.boxplot([clean_errors, dusty_errors], labels=['Clean', 'Dusty'])
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors by Category')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/figures/reconstruction_boxplot.png")

    print(" Evaluation results saved under results/metrics and results/figures.")

if __name__ == "__main__":
    evaluate_cleanliness_threshold()
