"""
test_mutamorphic.py
"""

import pytest
from joblib import load

# pylint: disable=redefined-outer-name


@pytest.fixture()
def model_setup():
    """
    Setup function to prepare the environment for testing
    Returns the model
    """
    joblib_output_dir = "tmp/"
    model = load(joblib_output_dir + "model.joblib")

    if model is None:
        raise ValueError("Model is not loaded correctly.")

    return model


# test inputs
@pytest.mark.parametrize(
    "original, transformed, label",
    [
        # ("The food was great.", "The food was amazing.", 1),
        ("The food was nasty.", "The food was really bad.", 0),
        # ("Waited too long.", "The wait was very long.", 0),
        # ("Staff was friendly.", "The staff was nice.", 1),
        # ("Pretty loud inside.", "Kinda noisy indoors.", 0),
        # ("Service was fast.", "Service was quick.", 1),
        # ("Good drink selection.", "Nice variety of drinks.", 1),
        # ("The pasta was bland.", "The pasta lacked flavor.", 0),
        # ("Amazing", "Terrible", 1),
    ]
)
def test_mutamorphic_review_consistency(model_setup, original, transformed, label):
    """
    Test to assess the model's robustness by ensuring the model performs similarly on semantically equivalent reviews
    Mutamorphic Testing: Comparing outputs of test inputs with context-similar alternatives (e.g. okay vs. ﬁne)
    """
    model = model_setup
    vectorizer = load("tmp/cv.joblib")

    test_inputs = vectorizer.transform([original, transformed]).toarray()
    preds = model.predict(test_inputs)

    # Inconsistent predictions
    assert preds[0] == preds[1], (
        f"Inconsistent predictions:\n"
        f"  Original:    {original} -> {preds[0]}\n"
        f"  Transformed: {transformed} -> {preds[1]}"
    )

    assert preds[0] == label, f"Original: '{original}' -> predicted {preds[0]}, expected {label}"
    assert preds[1] == label, f"Transformed: '{transformed}' -> predicted {preds[1]}, expected {label}"
