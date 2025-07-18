"""
test_monitoring.py
"""

import time
import tracemalloc
import pytest
from joblib import load

# pylint: disable=redefined-outer-name

MAX_MEMORY_MB = 10  # adjust
MAX_LATENCY_SECS = 0.5  # adjust


@pytest.fixture()
def monitoring_setup():
    """
    Setup function to prepare the environment for testing
    Returns the model and test data
    """
    joblib_output_dir = "tmp/"
    corpus = load(joblib_output_dir + "corpus.joblib")
    corpus_processed = load(joblib_output_dir + "corpus_processed.joblib")
    model = load(joblib_output_dir + "model.joblib")
    X = load(joblib_output_dir + "X.joblib")
    y = load(joblib_output_dir + "y.joblib")
    X_test = load(joblib_output_dir + "X_test.joblib")
    y_test = load(joblib_output_dir + "y_test.joblib")

    components = (corpus, corpus_processed, model, X, y, X_test, y_test)
    if any(component is None for component in components):
        raise ValueError("One or more components are not loaded correctly.")

    return components


def test_data_invariants(monitoring_setup):
    """
    Test to ensure data invariants are maintained.
    This includes checking that the corpus and labels are not empty, shape and length checks.
    !ML Test Score, Monitor 2: Data invariants hold in training and serving inputs
    """
    corpus, corpus_processed, model, X, y, X_test, y_test = monitoring_setup

    # Check if any component is None
    assert corpus is not None, "Corpus should not be None"
    assert corpus_processed is not None, "Processed corpus should not be None"
    assert model is not None, "Model should not be None"
    assert X is not None, "Features X should not be None"
    assert y is not None, "Labels should not be None"
    assert X_test is not None, "Test features X_test should not be None"
    assert y_test is not None, "Test labels y_test should not be None"

    # Shape checks
    assert X.shape[0] == len(y), "Number of samples in X should match number of labels y"
    assert X_test.shape[0] == len(
        y_test
    ), "Number of samples in X_test should match number of test labels y_test"
    assert X.shape[1] > 0, "Features X should have at least one feature"
    assert X_test.shape[1] > 0, "Test features X_test has to have at least one feature"
    assert (
        X.shape[1] == X_test.shape[1]
    ), "Features in X and X_test should have the same number of dimensions"

    # Length checks
    assert (len(corpus) > 0) and (len(corpus_processed) > 0), "Corpus cannot be empty"
    assert (len(y) > 0) and (len(y_test) > 0), "Labels cannot be empty"
    assert len(set(y)) == len(
        set(y_test)
    ), "Number of unique labels y has to match number of test labels y_test"
    assert len(set(y)) > 0, "Labels y should have at least one unique label"
    assert len(set(y_test)) > 0, "Test labels y_test have to have at least one unique label"
    assert len(corpus_processed) == len(
        y
    ), "Processed corpus has to have the same number of samples as labels y"

    # Check if model classes match unique labels in y
    assert set(model.classes_) == set(y), "Model classes do not match unique labels in y"


def test_prediction_memory_usage(monitoring_setup):
    """
    Test to ensure memory usage is within acceptable limits.
    Measures the peak memory usage during the model prediction of the test set.
    """
    _, _, model, _, _, X_test, _ = monitoring_setup

    tracemalloc.start()

    model.predict(X_test)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 ** 2

    assert peak_mb < MAX_MEMORY_MB, f"Memory usage too high: {peak_mb:.2f} MB (limit: {MAX_MEMORY_MB} MB)"


def test_prediction_latency(monitoring_setup):
    """
    Test to ensure model prediction latency is within acceptable limits.
    Measures the wall-clock time taken to run model.predict(X_test).
    """
    _, _, model, _, _, X_test, _ = monitoring_setup

    start_time = time.time()

    model.predict(X_test)

    end_time = time.time()
    latency_secs = end_time - start_time

    assert latency_secs < MAX_LATENCY_SECS, (
        f"Prediction latency too high: {latency_secs:.3f} s "
        f"(limit: {MAX_LATENCY_SECS} s)"
    )
