# General imports
import os
from joblib import dump
import datetime
import pandas as pd

# ML imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from lib_ml.preprocess import preprocess_text

def train(file: str = "data/RestaurantReviews_HistoricDump.tsv", max_features: int = 1420, test_size: float = 0.30, seed: int = 42):
    """
    Trains a sentiment analysis classifier.

    Parameters:
        - file: (str), path to dataset.
        - max_features: (int), number of features for CountVectorizer.
        - test_size: (float), train/test split ratio.
        - seed: (int), random seed.

    Returns:
        - tuple: (trained model, vectorizer)
    """
    print(os.getcwd())

    # Loading the dataset, preprocessing, splitting
    print("Loading dataset...")
    dataset = pd.read_csv(file, delimiter="\t", quoting=3)
    corpus = preprocess_text(dataset)
    y = dataset.iloc[:, -1].values
    
    print("Processing dataset...")
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Train the model
    print("Training model...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred =  model.predict(X_test)

    # Evaluate performance
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)

    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    now = datetime.datetime.now()
    timestamped_version = now.strftime("%Y%m%d")

    os.makedirs("tmp",exist_ok=True)
    dump(model, f"tmp/model-{timestamped_version}.pkl")
    dump(cv, f"tmp/bow-{timestamped_version}.pkl")

if __name__ == "__main__":
    train()    
