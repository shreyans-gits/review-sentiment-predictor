import joblib
from data.preprocess import clean_text

def load_model():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("data/vectorizer.pkl")
    return model,vectorizer

def predict(review, model, vectorizer):
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(vectorized_review)
    
    if prediction[0] == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"
    
def main():
    model, vectorizer = load_model()
    review = ""
    while review != "quit":
        review = input("Enter a review (or 'quit' to exit): ")
        if review != "quit":
            result = predict(review, model, vectorizer)
            print(result)

if __name__ == "__main__":
    main()