# General imports
import os
from joblib import dump

# ML imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def train(file : str ="/data/RestaurantReviews_HistoricDump.tsv", max_features : int = 1420, test_size : float =0.20, seed : int =42) -> None:

    # Needs to be connected through model-service to ml-lib for the preprocessing
    X, y = ml_lib.preprocess(file)
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(X).toarray()
    
    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Train classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred =  model.predict(X_test)

    # Evaluate performance
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cm)
    print(acc)

    # Save BoW dictionary & trained model
    os.makedirs("model", exist_ok=True)
    dump(cv, "model/bow.pkl")
    dump(model, "model/model.pkl")



if __name__ == "__main__":
    train()    
