import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#loads the pkl file
def load_artifacts():
    X_train = joblib.load("data/X_train.pkl")
    X_test = joblib.load("data/X_test.pkl")
    y_train = joblib.load("data/y_train.pkl")
    y_test = joblib.load("data/y_test.pkl")
    return X_train, X_test, y_train, y_test

#creates a model and trains it
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

#evaluates the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{matrix}")

#saves the model
def save_model(model):
    joblib.dump(model,"model/model.pkl")

def main():
    X_train, X_test, y_train, y_test = load_artifacts()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    
if __name__ == "__main__":
    main()