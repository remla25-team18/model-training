"""
test_infrastructure.py
"""

import pytest
import os
from joblib import load, dump
from src.modeling.train import train
from src.modeling.evaluate import evaluate


@pytest.fixture()
def paths_setup():
    """
    Setup function to prepare the environment for testing
    Returns the directories for joblib and JSON outputs
    """
    joblib_output_dir = "tmp/"
    json_output_dir = "metrics/"

    return joblib_output_dir, json_output_dir


def test_if_files_saved_correctly(paths_setup):
    """
    Check if the necessary files are saved correctly in the specified directories.
    """

    joblib_output_dir, json_output_dir = paths_setup
    # Check if all joblib files exist
    joblib_files = [
        "model.joblib",
        "corpus.joblib",
        "cv.joblib",
        "X.joblib",
        "y.joblib",
        "X_test.joblib",
        "y_test.joblib",
    ]
    for file in joblib_files:
        assert os.path.isfile(
            os.path.join(joblib_output_dir, file)
        ), f"{file} not found in {joblib_output_dir}"

    # Check if metrics file exists
    json_file = "metrics.json"
    assert os.path.isfile(
        os.path.join(json_output_dir, json_file)
    ), f"{json_file} not found in {json_output_dir}"


def test_reproducibility(paths_setup):
    """
    Check that running the model training multiple times produces the same results.
    ML Test Score Infra 1: Training is reproducible.
    """
    joblib_output_dir, json_output_dir = paths_setup

    # Run the training function
    train(joblib_output_dir=joblib_output_dir, model_output_dir="./models/")

    # Load the model and data
    model = load(os.path.join(joblib_output_dir, "model.joblib"))
    X_test = load(os.path.join(joblib_output_dir, "X_test.joblib"))
    y_test = load(os.path.join(joblib_output_dir, "y_test.joblib"))
    print("infra", y_test)

    # Run the evaluation function
    evaluate(joblib_output_dir=joblib_output_dir, json_output_dir=json_output_dir)

    # Check if the model predictions are consistent
    y_pred_1 = model.predict(X_test)
    assert len(y_pred_1) == len(y_test), "Predictions length does not match test labels length"
    assert all(
        pred in set(y_test) for pred in y_pred_1
    ), "Some predictions are not in the set of unique labels"

    # Repeat the training and evaluation
    train(joblib_output_dir=joblib_output_dir, model_output_dir="./models/")
    model = load(os.path.join(joblib_output_dir, "model.joblib"))
    X_test = load(os.path.join(joblib_output_dir, "X_test.joblib"))
    y_test = load(os.path.join(joblib_output_dir, "y_test.joblib"))
    evaluate(joblib_output_dir=joblib_output_dir, json_output_dir=json_output_dir)
    y_pred_2 = model.predict(X_test)
    assert all(
        pred in set(y_test) for pred in y_pred_2
    ), "Some predictions are not in the set of unique labels"

    # Check if predictions are the same
    assert (y_pred_1 == y_pred_2).all(), "Model predictions are not reproducible across runs"
