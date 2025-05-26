"""
get_data.py
"""

import os
from joblib import dump
import pandas as pd
from lib_ml.preprocess import preprocess_text
import gdown


def download_file_from_google_drive(training_file_path):
    """
    Downloads a file from google drive
    """
    gdrive_url = "https://drive.google.com/uc?id=12ibFg0ReSlJYkTpPFbOhkzvZ1TN5bciO"
    gdown.download(gdrive_url, training_file_path, quiet=False)


def get_data(training_file_path, joblib_output_dir):
    """
    Loads the dataset from local folder, preprocesses it and saves it in joblib format

    Parameters:
        - training_file_path: str, path to the training file
        - joblib_output_dir: str, output directory for joblib files
    
    Returns:
        - corpus: list, preprocessed text data
        - y: numpy array, labels of the dataset
    """
    dataset = pd.read_csv(training_file_path, delimiter="\t", quoting=3)
    corpus = preprocess_text(dataset)
    y = dataset.iloc[:, -1].values

    dump(corpus, joblib_output_dir + "corpus.joblib")
    dump(y, joblib_output_dir + "y.joblib")


if __name__ == "__main__":
    os.makedirs("tmp", exist_ok=True)
    download_file_from_google_drive("data/raw/RestaurantReviews_HistoricDump.tsv")
    get_data("data/raw/RestaurantReviews_HistoricDump.tsv", "tmp/")
    print("Data saved to data and tmp directory.")
