"""
test_data.py
"""
import pytest
from joblib import load

@pytest.fixture()
def data_setup():
    """
    Setup function to prepare the environment for testing
    Returns the corpus and labels
    """
    corpus = load("tmp/corpus.joblib")
    y = load("tmp/y.joblib")
    if corpus is None or y is None:
        raise ValueError("Corpus or labels are not loaded correctly.")
    return corpus, y


def test_code_is_tested():
    """
    dummy function for now
    """
    # assert False
    assert True

def test_correct_format(data_setup):
    """
    Test to ensure there are no missing values in the dataset
    """
    corpus, y = data_setup
    assert corpus is not None, "Corpus should not be None"
    assert y is not None, "Labels should not be None"
    assert len(corpus) > 0, "Corpus should not be empty"
    assert len(y) > 0, "Labels should not be empty"

def test_no_missing_values(data_setup):
    """
    Test to ensure there are no missing values in the dataset
    """
    corpus, y = data_setup
    # Check for missing values
    for i in range(len(corpus)):
        # print(f"{i}: {corpus[i]}")
        if not corpus[i]:
            print(f"Missing value found in corpus at index {i}")
    assert all(corpus), "Corpus contains missing values"
    assert all(y), "Labels contain missing values"