from joblib import dump
import pandas as pd
from lib_ml.preprocess import preprocess_text


def get_data(training_file_dir, joblib_output_dir):
    # Loading the dataset
    dataset = pd.read_csv(training_file_dir, delimiter="\t", quoting=3)
    corpus = preprocess_text(dataset)
    y = dataset.iloc[:, -1].values

    dump(corpus, joblib_output_dir + "corpus.joblib")
    dump(y, joblib_output_dir + "y.joblib")


if __name__ == "__main__":
    get_data("data/raw/RestaurantReviews_HistoricDump.tsv", "tmp/")
    print("Data saved to tmp directory.")
