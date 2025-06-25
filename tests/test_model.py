"""
test_model.py
"""

import pytest
import numpy as np
from joblib import load

# pylint: disable=redefined-outer-name


@pytest.fixture()
def model_and_data_setup():
    """
    Setup function to prepare the environment for testing
    Returns the model, X_test, and y_test
    """
    joblib_output_dir = "tmp/"
    model = load(joblib_output_dir + "model.joblib")
    X_test = load(joblib_output_dir + "X_test.joblib")
    y_test = load(joblib_output_dir + "y_test.joblib")

    if model is None or X_test is None or y_test is None:
        raise ValueError("Model or test data is not loaded correctly.")

    return model, X_test, y_test


def test_model_performance(model_and_data_setup):
    """
    Test to ensure the model performs as expected
    !ML Test Score, Model 5: A simpler model is not better
    """
    model, X_test, y_test = model_and_data_setup
    y_pred = model.predict(X_test)

    # Baseline accuracy according the the majority class in y_test
    baseline_accuracy = np.unique(y_test, return_counts=True)[1].max() / len(y_test)
    print(f"Baseline accuracy: {baseline_accuracy}")

    # Check if predictions are of the same length as y_test
    assert len(y_pred) == len(y_test), "Predictions length does not match test labels length"

    # Check if predictions are not empty
    assert len(y_pred) > 0, "Predictions should not be empty"

    # Check if all predictions are in the range of labels
    unique_labels = set(y_test)
    for pred in y_pred:
        assert (
            pred in unique_labels
        ), f"Prediction {pred} is not in the set of unique labels {unique_labels}"

    # Check if accuracy is above a baseline
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy}")
    assert (
        accuracy >= baseline_accuracy
    ), f"Model accuracy {accuracy} is below baseline {baseline_accuracy}"
