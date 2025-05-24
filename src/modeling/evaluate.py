'''
evaluate.py
'''

import json
import os
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate(joblib_output_dir, json_output_dir):
    '''
    Evaluates the model
    '''
    model = load(joblib_output_dir + "model.joblib")
    X_test = load(joblib_output_dir + "X_test.joblib")
    y_test = load(joblib_output_dir + "y_test.joblib")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate performance
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Confusion_matrix: ", cm)
    print("Accuracy: ", acc)

    # Save metrics to JSON
    metrics = {
        "confusion_matrix": cm.tolist(),  # Convert numpy array to list for JSON serialization
        "accuracy": acc,
    }
    os.makedirs(json_output_dir, exist_ok=True)
    with open(os.path.join(json_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    evaluate(
        joblib_output_dir="tmp/",
        json_output_dir="metrics/",
    )
