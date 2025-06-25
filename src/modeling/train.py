"""
train.py
"""
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from joblib import load, dump


def train(joblib_output_dir, model_output_dir, test_size: float = 0.30, seed: int = 42):
    """
    Splits the dataset, trains the model and saves it
    """

    X = load(joblib_output_dir + "X.joblib")
    y = load(joblib_output_dir + "y.joblib")
    cv = load(joblib_output_dir + "cv.joblib")

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Train the model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Store the model in joblib
    dump(model, joblib_output_dir + "model.joblib")
    dump(X_test, joblib_output_dir + "X_test.joblib")
    dump(y_test, joblib_output_dir + "y_test.joblib")
    dump(cv, joblib_output_dir + "cv.joblib")

    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    now = datetime.datetime.now()
    timestamped_version = now.strftime("%Y%m%d")

    # Save the model
    os.makedirs(model_output_dir, exist_ok=True)
    dump(model, os.path.join(model_output_dir, f"model-{timestamped_version}.pkl"))
    dump(cv, os.path.join(model_output_dir, f"cv-{timestamped_version}.pkl"))


if __name__ == "__main__":
    train(
        joblib_output_dir="tmp/",
        model_output_dir="./models/",
    )
