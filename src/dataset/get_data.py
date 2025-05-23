'''
get_data.py
'''

from joblib import dump
import pandas as pd
from lib_ml.preprocess import preprocess_text
import gdown


def download_file_from_google_drive(training_file_dir):
    '''
    Downloads a file from google drive
    '''
    gdrive_url = "https://drive.google.com/uc?id=12ibFg0ReSlJYkTpPFbOhkzvZ1TN5bciO"
    gdown.download(gdrive_url, training_file_dir, quiet=False)


def get_data(training_file_dir, joblib_output_dir):
    '''
    Loads the dataset
    '''
    dataset = pd.read_csv(training_file_dir, delimiter="\t", quoting=3)
    corpus = preprocess_text(dataset)
    y = dataset.iloc[:, -1].values

    dump(corpus, joblib_output_dir + "corpus.joblib")
    dump(y, joblib_output_dir + "y.joblib")


if __name__ == "__main__":
    download_file_from_google_drive("data/raw/RestaurantReviews_HistoricDump.tsv")
    get_data("data/raw/RestaurantReviews_HistoricDump.tsv", "tmp/")
    print("Data saved to tmp directory.")
