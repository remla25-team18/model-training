# Model training

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

4. Create `tmp` folder under the root directory of the project. This folder will be used to store joblib output during the stages.

    ```bash
    mkdir tmp
    ```

5. Run dvc using `dvc repro`, you should see the following output:

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

6. [**GDrive secret needed**] You can also push and pull the data to and from the remote storage(Google Drive) using:

    ```bash
    dvc push
    dvc pull
    ```


## Instructions on how to set up remote storage

In `./src/get_data.py`, you can set up remote storage for DVC. We used Google Drive as remote storage.

The link to the drive folder is automatically set and is open to everyone.
