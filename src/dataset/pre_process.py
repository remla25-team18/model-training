"""
pre_process.py
"""

from joblib import load, dump
import numpy as np

# ML imports
from sklearn.feature_extraction.text import CountVectorizer


def remove_missing_values(corpus, y):
    """
    Removes missing values from the corpus-label pairs.
    Parameters:
        corpus (list): List of text data.
        y (list): List of labels corresponding to the text data.
    Returns:
        list, list: Lists of text data with missing values removed.
    """
    cleaned_corpus = []
    cleaned_y = []
    for text, label in zip(corpus, y):
        if text != "" and label is not None:  # Check if both text and label are not empty
            # Check if both text and label are not empty
            cleaned_corpus.append(text)
            cleaned_y.append(label)
    return cleaned_corpus, np.array(cleaned_y)


def preprocess(
    joblib_output_dir,
    max_features: int = 1420,
):
    """
    Pre-processes the data
    """
    # Load data from joblib
    corpus = load(joblib_output_dir + "corpus.joblib")
    y = load(joblib_output_dir + "raw_labels.joblib")
    print(type(corpus), type(y))

    # Remove missing values
    corpus, y = remove_missing_values(corpus, y)

    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()

    # Store the data in joblib
    dump(corpus, joblib_output_dir + "corpus_processed.joblib")
    dump(X, joblib_output_dir + "X.joblib")
    dump(y, joblib_output_dir + "y.joblib")
    dump(cv, joblib_output_dir + "cv.joblib")


if __name__ == "__main__":
    preprocess("tmp/")
    print("Data saved to tmp directory.")
