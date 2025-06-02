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
    ML Test Score, Model 5: A simpler model is not better
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

# First and last 4 reviews from the dataset
@pytest.mark.parametrize(
    "original, transformed, label",
    [
        ("The food was great.", "The food was amazing.", 1),
        ("A bit too salty.", "Slightly too salty.", 0),
        ("Waited too long.", "The wait was very long.", 0),
        ("Staff was friendly.", "The staff was nice.", 1),
        ("Pretty loud inside.", "Kinda noisy indoors.", 0),
        ("Service was fast.", "Service was quick.", 1),
        ("Good drink selection.", "Nice variety of drinks.", 1),
        ("The pasta was bland.", "The pasta lacked flavor.", 0),
        ("Amazing", "Terrible", 1),
    ]
)

def test_metamorphic_review_consistency(model_and_data_setup, original, transformed, label):
    """
    Test to assess the robustness of the model by ensuring the model performs similarly on semantically equivalent reviews
    Mutamorphic Testing: Comparing outputs of test inputs with context-similar alternatives (synonyms, e.g. okay vs. ﬁne)
    """

    model, X_test, y_test = model_and_data_setup
    vectorizer = load("tmp/cv.joblib")

    X_vec = vectorizer.transform([original, transformed]).toarray()
    preds = model.predict(X_vec)

    # Inconsistent predictions
    assert preds[0] == preds[1], (
        f"Inconsistent predictions:\n"
        f"  Original:    {original} -> {preds[0]}\n"
        f"  Transformed: {transformed} -> {preds[1]}"
    )

    # assert preds[0] == label, f"Incorrect prediction for original: '{original}' -> predicted {preds[0]}, expected {label}"
    # assert preds[1] == label, f"Incorrect prediction for transformed: '{transformed}' -> predicted {preds[1]}, expected {label}"

