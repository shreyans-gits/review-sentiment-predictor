import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

stop_words = set(stopwords.words('english')) #removes auxilary words

#reads and loads dataset from csv
def load_data():
    df = pd.read_csv("data/IMDB Dataset.csv")
    X = df['review']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    return X, y

#covert text into lower case, remove space
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    text = ' '.join(words)
    return text

#append text in cleaned array
def clean_dataset(texts):
    cleaned = []
    for text in texts:
        cleaned.append(clean_text(text))
    return cleaned

#80% is for learning and 20% is for testing
def split_data(texts, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test

#creates a list of numbers that the model can understand
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer

#it stores what it learnt from the dataset in the memory
def save_artifacts(vectorizer,X_train,X_test,y_train,y_test):
    joblib.dump(vectorizer,"data/vectorizer.pkl")
    joblib.dump(X_train,"data/X_train.pkl")
    joblib.dump(X_test,"data/X_test.pkl")
    joblib.dump(y_train,"data/y_train.pkl")
    joblib.dump(y_test,"data/y_test.pkl")

def main():
    print("Loading data...")
    X, y = load_data()

    print("Cleaning data...")
    X_cleaned = clean_dataset(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_cleaned, y)

    print("Vectorizing...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize(X_train, X_test)

    print("Saving artifacts...")
    save_artifacts(vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test)

    print("Done! All artifacts saved to data/")

if __name__ == "__main__":
    main()