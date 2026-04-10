# 🎬 Movie Review Sentiment Classifier

A machine learning project that classifies movie reviews as positive or negative using Natural Language Processing (NLP) and Logistic Regression.

---

## 📊 Model Performance

- **Accuracy:** 88.85%
- **Dataset:** IMDB Movie Reviews (50,000 reviews)
- **Algorithm:** Logistic Regression with TF-IDF Vectorization

---

## 📁 Project Structure

sentiment-classifier/
├── data/
│   ├── preprocess.py       # Data loading, cleaning, vectorizing
│   └── *.pkl               # Generated artifacts (not tracked by git)
├── model/
│   ├── train.py            # Model training and evaluation
│   └── model.pkl           # Trained model (not tracked by git)
├── predict.py              # Interactive prediction script
├── requirements.txt        # Dependencies
└── README.md

---

## 🛠️ Setup

### 1. Clone the repo
git clone https://github.com/yourusername/sentiment-classifier.git
cd sentiment-classifier

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

### 4. Download the dataset
Download the IMDB Dataset from Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place it in the data/ folder as "IMDB Dataset.csv"

---

## 🚀 Running the Project

### Step 1 — Preprocess the data
python data/preprocess.py

This will generate all .pkl files in the data/ folder.

### Step 2 — Train the model
python model/train.py

This will train the model and save it to model/model.pkl

### Step 3 — Run predictions
python predict.py

---

## 💬 Example Usage

Enter a review (or 'quit' to exit): This movie was absolutely amazing, I loved every second of it!
Positive 😊

Enter a review (or 'quit' to exit): Worst film I have ever seen, complete waste of time.
Negative 😞

Enter a review (or 'quit' to exit): quit

---

## 🔧 How It Works

1. **Load** — 50,000 IMDB movie reviews are loaded from CSV
2. **Clean** — Text is lowercased, HTML tags removed, punctuation stripped, stopwords removed
3. **Split** — Data is split 80% train / 20% test with stratification
4. **Vectorize** — TF-IDF converts text to numerical features (top 5000 words)
5. **Train** — Logistic Regression model is trained on the data
6. **Evaluate** — Model achieves 88.85% accuracy on unseen reviews
7. **Predict** — User can input any review and get instant sentiment prediction

---

## 📦 Dependencies

- scikit-learn
- pandas
- numpy
- nltk
- joblib
- matplotlib
- seaborn
- wordcloud

---

## 👤 Authors

Tanisha : tanisha2607
Shreyans : shreyans-gits