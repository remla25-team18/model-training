# Model training

## Continuous Integration Metrics

The following metrics are automatically updated with each push via GitHub Actions:
The pytests and coverae report can also be done manually. For the relevant commands, first following the instructions below and then run the command found in the workflow.

| Metric | Badge |
|--------|-------|
| **Test Coverage** | ![coverage](coverage.svg) |
| **Pylint Score** | ![pylint-score](pylint.svg) * |

* work in progress

## Installation

1. Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt) for a lightweight Python environment according to your OS:

    - During installation, **type "yes" when prompted to initialize Miniconda in your shell** (this adds initialization to `.bashrc` or `.bash_profile`).
    - After installation, reload your shell configuration:

      ```bash
      source ~/.bashrc
      ```

    - Verify the installation:

      ```bash
      conda --version
      ```

      You should see the installed conda version.

2. Create and activate a new conda environment with Python 3.11:

    ```bash
    conda create -n ml python=3.11
    conda activate ml
    ```

3. Install the required packages using pip:

    ```bash
    pip install -e .
    ```

4. Run dvc using `dvc repro`, you should see the following output:

    ```plaintext
    Running stage 'get_data':    
    Added stage 'get_data' in 'dvc.yaml'
    Updating lock file 'dvc.lock'                           

    Running stage 'data_preprocess':                                                       
    > python ./src/data/pre_process.py
    Data saved to tmp directory.
    Updating lock file 'dvc.lock'                                                          

    Running stage 'train':                                                                 
    > python ./src/model/train.py
    Saving model and vectorizer...
    Updating lock file 'dvc.lock'                                                          

    Running stage 'evaluate':                                                              
    > python ./src/model/evaluate.py
    Confusion_matrix:  [[ 64  60]
    [ 19 127]]
    Accuracy:  0.7074074074074074
    ```

    The metrics are saved in `metrics.json` file under the `metrics/` directory of the project, you can check the metrics using:

    ```bash
    dvc metrics show
    ```

5. [**GDrive secret needed**] You can also push and pull the data to and from the remote storage(Google Drive) using:

    ```bash
    dvc push
    dvc pull
    ```

## Instructions on how to set up remote storage

In `./src/get_data.py`, you can set up remote storage for DVC. We used Google Drive as remote storage.

The link to the drive folder is automatically set and is open to everyone.

## Code Quality

1. Pylint has a non-standard configuration, which can be checked in pylintrc, and one custom rule for the ML code smell Randomness Uncontrolled, which can be checked in linter_rules/pylint. To run pylint use:

    ```bash
    pylint .
    ```
Ideal output:

    Your code has been rated at 10.00/10

2. Flake8 had a non-default configuration, which can be checked in linter_rules/flake8. To run flake8 use:

    ```bash
    pip install -e linter_rules/flake8 --use-pep517
    flake8 . --verbose
    ```

Ideal output:

    flake8.checker            MainProcess     89 INFO     Making checkers
    flake8.main.application   MainProcess    136 INFO     Finished running
    flake8.main.application   MainProcess    136 INFO     Reporting errors
    flake8.main.application   MainProcess    137 INFO     Found a total of 0 violations and reported 0'

3. Bandit has a non-default configuration, which can be checked in bandit.yaml. To run bandit use:
    
    ```bash
    bandit -r .
    ```

Ideal output: 

    [main]  INFO    profile include tests: None
    [main]  INFO    profile exclude tests: None
    [main]  INFO    cli include tests: None
    [main]  INFO    cli exclude tests: None
    [main]  INFO    running on Python 3.11.11
    Run started:2025-05-25 20:55:57.477128

    Test results:
            No issues identified.

    Code scanned:
            Total lines of code: 203
            Total lines skipped (#nosec): 0

    Run metrics:
            Total issues (by severity):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
            Total issues (by confidence):
                    Undefined: 0
                    Low: 0
                    Medium: 0
                    High: 0
    Files skipped (0):

All three linters are automatically run as part of the GitHub workflow.