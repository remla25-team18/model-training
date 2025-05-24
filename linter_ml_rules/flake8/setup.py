'''
setup.py
'''
from setuptools import setup

setup(
    name="flake8-dataset-split-kwarg-checker",
    py_modules=["check_kwarg_dataset_split"],
    entry_points={
        "flake8.extension": [
            "TTS001 = check_kwarg_dataset_split:DatasetSplitKwargChecker",
        ],
    },
)
