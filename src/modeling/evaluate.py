from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score


def evaluate(joblib_output_dir):
    model = load(joblib_output_dir + "model.joblib")
    X_test = load(joblib_output_dir + "X_test.joblib")
    y_test = load(joblib_output_dir + "y_test.joblib")

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate performance
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Confusion_matrix: ", cm)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    evaluate(
        joblib_output_dir="tmp/",
    )
