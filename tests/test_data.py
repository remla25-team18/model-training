"""
test_data.py
"""

import pytest
import numpy as np
from joblib import load

# pylint: disable=redefined-outer-name


@pytest.fixture()
def data_setup():
    """
    Setup function to prepare the environment for testing
    Returns the corpus and labels
    """
    corpus = load("tmp/corpus_processed.joblib")
    y = load("tmp/y.joblib")
    if corpus is None or y is None:
        raise ValueError("Corpus or labels are not loaded correctly.")
    return corpus, y


@pytest.fixture()
def data_test_setup():
    """
    Additional setup function to get test data
    """
    y_test = load("tmp/y_test.joblib")
    if y_test is None:
        raise ValueError("Test labels are not loaded correctly.")
    return y_test


def test_correct_format(data_setup):
    """
    Test to ensure there are no missing values in the dataset
    !ML Test Score, Data 4: Features adhere to meta-level requirements
    """
    corpus, y = data_setup
    assert corpus is not None, "Corpus should not be None"
    assert y is not None, "Labels should not be None"
    assert len(corpus) > 0, "Corpus should not be empty"
    assert len(y) > 0, "Labels should not be empty"


def test_no_missing_values(data_setup):
    """
    Test to ensure there are no missing values in the dataset
    !ML Test Score, Data 4: Features adhere to meta-level requirements
    """
    corpus, y = data_setup
    # Check for missing values
    for i, doc in enumerate(corpus):
        if not doc:
            print(f"Missing value found in corpus at index {i}")
    assert all(corpus), "Corpus contains missing values"
    assert None not in y, "Labels contain missing values"


def test_labels_in_the_test_distributed_proportionally(data_setup, data_test_setup):
    """
    Test to ensure labels are distributed proportionally in the overall and test datasets
    !ML Test Score, Data 1: Feature expectations are captured in a schema
    """
    _, y = data_setup
    y_test = data_test_setup
    total_count = len(y)
    total_count_test = len(y_test)
    unique_labels, counts = np.unique(y, return_counts=True)
    unique_labels_test, counts_test = np.unique(y_test, return_counts=True)
    # Check if all labels are present
    assert len(unique_labels) > 0, "No labels found in the dataset"

    # Check if the labels in the test set are a subset of the overall labels
    assert set(unique_labels_test).issubset(
        set(unique_labels)
    ), "Test labels are not a subset of overall labels"

    # Check if the distribution of labels in the overall dataset is not too skewed
    min_count_prop = min(counts) / total_count
    print(f"Label proportion difference in overall set: \n{counts / total_count}")
    assert min_count_prop > 0.2, "Labels are not distributed proportionally in the overall dataset"

    # Check if the distribution is proportional in the test set
    count_diff_test = (max(counts_test) - min(counts_test)) / total_count_test
    print(f"Label proportion difference in test set: \n{counts_test / total_count_test}")
    assert count_diff_test < 0.2, "Test labels are not distributed proportionally"
