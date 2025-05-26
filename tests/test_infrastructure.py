"""
test_infrastructure.py
"""
import pytest

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
    import os

    joblib_output_dir, json_output_dir = paths_setup
    # Check if all joblib files exist
    joblib_files = ["model.joblib", "corpus.joblib", "cv.joblib", "X.joblib", "y.joblib", "X_test.joblib", "y_test.joblib"]
    for file in joblib_files:
        assert os.path.isfile(os.path.join(joblib_output_dir, file)), f"{file} not found in {joblib_output_dir}"

    # Check if metrics file exists
    json_file = "metrics.json"
    assert os.path.isfile(os.path.join(json_output_dir, json_file)), f"{json_file} not found in {json_output_dir}"