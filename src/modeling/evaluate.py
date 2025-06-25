"""
evaluate.py
"""

import json
import os
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def evaluate(joblib_output_dir, json_output_dir):
    """
    Evaluated the model
    """

    model = load(joblib_output_dir + "model.joblib")
    X_test = load(joblib_output_dir + "X_test.joblib")
    y_test = load(joblib_output_dir + "y_test.joblib")

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("Confusion_matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs(json_output_dir, exist_ok=True)
    with open(os.path.join(json_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    evaluate(
        joblib_output_dir="tmp/",
        json_output_dir="metrics/",
    )
