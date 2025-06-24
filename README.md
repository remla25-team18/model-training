# Restaurant Sentiment Classification – ML Configuration and Testing

This repository implements an end-to-end machine learning pipeline for sentiment classification using the [Cookiecutter Data Science template](https://github.com/drivendataorg/cookiecutter-data-science). It adheres to best practices in configuration management and ML testing, and supports reproducibility through [DVC](https://dvc.org/), [GitHub Actions](https://docs.github.com/en/actions), and rigorous automated testing based on the [ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/).

---
| Metric            | Badge                    |
|------------------|--------------------------|
| Test Coverage     | ![coverage](coverage.svg) |
| Pylint Score      | ![pylint-score](pylint.svg) |
---

## 1. Project Setup and Installation

### Conda Environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and initialize it in your shell:
    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

2. Create a new environment:
    ```bash
    conda create -n ml python=3.11
    conda activate ml
    ```

3. Install dependencies:
    ```bash
    pip install -e .
    ```

---

## 2. Running the Pipeline with DVC

The ML pipeline is managed with DVC and consists of the following stages:
- `get_data`: download and preprocess raw data
- `data_preprocess`: vectorize and clean the corpus
- `train`: train a Naive Bayes classifier
- `evaluate`: evaluate and output performance metrics

Run the full pipeline with:
```bash
dvc repro
```

Example output:
```
Running stage 'get_data':    
> python ./src/data/get_data.py
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
Confusion_matrix:  [[ 79  47]
                    [ 27 117]]
Accuracy:  0.725925925925926
```

To inspect results:
```bash
dvc metrics show
```

Try different experiments by modifying parameters in the `train` stage inside `dvc.yaml` and re-running the pipeline. DVC will track changes and allow you to compare results.

After chaning it, re-run the pipeline to see updated metrics:
```bash
dvc exp run
dvc exp show
```
You will see output like:

| Experiment             | Created  | Accuracy |
|------------------------|----------|----------|
| workspace              | –        | 0.7037   |
| lemon/a4_fix           | 07:17 PM | 0.68148  |
| └── 9092e4d [brisk-cedi] | 07:27 PM | 0.7037   |


---

## 3. Google Drive Remote Storage

#### Remote Data Storage
This project uses **public Google Drive remote** to version datasets and models. No authentication or secrets are required to access the data.

> The pipeline uses a public Google Drive folder set in `src/dataset/get_data.py`.

#### DVC Versioning 
If you want to explore the dvc versioning system, you need to configure remote storage of Google Drive. The registration process can be found in [Set up google drive auth](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended). After setting up, you can contact the admin(lemonhe@tudelft.nl) to add your account to the trust list. Then, you can add the remote storage to your DVC project:

```bash
dvc remote modify gdrive gdrive_client_id [id]
dvc remote modify gdrive gdrive_client_secret [secret]
```

To push/pull artifacts, use:

```bash
dvc push
dvc pull
```

You will see output like:
```
(test) model-training % dvc pull
Collecting  
|0.00 [00:00,    ?entry/s]
Fetching
Building workspace index 
12.0 [00:00,  966entry/s]
Comparing indexes
13.0 [00:00, 5.61kentry/s]
Applying changes
0.00 [00:00,     ?file/s]
Everything is up to date.

(test) model-training % dvc push
Collecting
0.00 [00:00,    ?entry/s]
Pushing
5 files pushed                                       
```


---

## 4. Code Quality and Linting

We use three linters with customized rules:
- **Pylint**: configured in `pylintrc` with custom ML code smell rules
- **Flake8**: configured in `linter_rules/flake8`
- **Bandit**: configured in `bandit.yaml` for security scanning

Run them via:
```bash
pylint .
flake8 . --verbose
bandit -r . -c bandit.yaml
```

All linters are integrated into the CI pipeline.

---

## 5. Automated Testing

Our test suite follows the ML Test Score methodology and includes tests for:

- **Feature and Data Integrity**: `tests/test_data.py`
- **Model Development**: `tests/test_model.py`
- **ML Infrastructure**: `tests/test_infrastructure.py`
- **Monitoring** (latency, memory): `tests/test_monitoring.py`
- **Metamorphic Testing**: `tests/test_mutamorphic.py` (semantic equivalence checks)

Run all tests with coverage:
```bash
pytest -v --cov=src --cov-report=xml --cov-report=term-missing tests/ | tee pytest-coverage.txt
```

---

## 6. Continuous Integration and Metrics Reporting

GitHub Actions automatically run:
- Linting (Pylint, Flake8, Bandit)
- Pytest with coverage
- Updates to README metrics badges

---

## 7. Exploratory Code

An exploratory notebook is stored in the `notebooks/` directory and kept separate from production code under `src/`. You can run it interactively to explore the dataset, visualize vocabulary features, and inspect misclassifications.

---

## 8. Directory Structure

```
├── .dvc/                     # DVC internal files
├── .github/workflows/        # GitHub Actions CI configuration
├── data/                     # Raw and processed data (DVC-tracked)
├── linter_rules/             # Custom linter rules (e.g. flake8 config)
├── notebooks/                # Jupyter notebooks for exploratory analysis
├── src/                      # Source code for data processing and modeling
│   ├── dataset/              # Data loading and preprocessing scripts
│   └── modeling/             # Model training and evaluation logic
├── tests/                    # Test suite following ML Test Score categories
├── .dvcignore                # DVC ignore file
├── .gitignore                # Git ignore file
├── Makefile                  # Commands for reproducible execution
├── README.md                 # Project documentation
├── bandit.yaml               # Bandit configuration for security linting
├── coverage.svg              # Auto-generated test coverage badge
├── dvc.lock                  # DVC lock file for pipeline reproducibility
├── dvc.yaml                  # DVC pipeline definition
├── pylintrc                  # Pylint configuration (non-standard)
├── pyproject.toml            # Project metadata and formatting tools
├── pytest-coverage.txt       # Output of pytest coverage report
├── setup.cfg                 # Additional config for tools (e.g., pytest)
```
