"""
pre_process.py
"""

from joblib import load, dump

# ML imports
from sklearn.feature_extraction.text import CountVectorizer


def preprocess(
    joblib_output_dir,
    max_features: int = 1420,
):
    """
    Pre-processes the data
    """
    # Load data from joblib
    corpus = load(joblib_output_dir + "corpus.joblib")
    y = load(joblib_output_dir + "y.joblib")

    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()

    # Store the data in joblib
    dump(X, joblib_output_dir + "X.joblib")
    dump(y, joblib_output_dir + "y.joblib")
    dump(cv, joblib_output_dir + "cv.joblib")


if __name__ == "__main__":
    preprocess("tmp/")
    print("Data saved to tmp directory.")
