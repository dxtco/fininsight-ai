# ml_pipeline.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Download VADER lexicon data
nltk.download('vader_lexicon', quiet=True)

def clean_text(text):
    """Cleans text data by lowering case and removing punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def process_and_train():
    print("Step 1: Ingesting a subset of Kaggle dataset...")
    # Load 20,000 rows to ensure we get a solid balance of data
    df = pd.read_csv('analyst_ratings_processed.csv', nrows=20000)
    
    print("Step 2: Cleaning text data...")
    df['cleaned_title'] = df['title'].apply(clean_text)
    
    print("Step 3: Algorithmic Labeling using VADER Sentiment Analysis...")
    sia = SentimentIntensityAnalyzer()
    
    labels = []
    for title in df['cleaned_title']:
        score = sia.polarity_scores(title)['compound']
        if score >= 0.05:
            labels.append(1)  # Bullish / Positive
        elif score <= -0.05:
            labels.append(0)  # Bearish / Negative
        else:
            labels.append(2)  # Neutral
            
    df['label'] = labels
    
    # Filter out neutral rows to make it a sharp binary classification task (Up vs Down)
    df = df[df['label'] != 2]
    
    print(f"Dataset Balanced. Class distribution:\n{df['label'].value_counts()}")

    # Features and Target
    X = df['cleaned_title']
    y = df['label']
    
    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Step 4: Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1500)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Step 5: Training Traditional ML Classifier (Random Forest)...")
    # Using Random Forest to demonstrate core ensemble learning knowledge
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_vec, y_train)
    
    print("Step 6: Evaluating Model Performance...")
    y_pred = model.predict(X_test_vec)
    
    print("\n================ MODEL EVALUATION METRICS ================")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bearish (0)', 'Bullish (1)']))
    print("==========================================================")
    
    return model, vectorizer

if __name__ == "__main__":
    import joblib
    
    model, vectorizer = process_and_train()
    
    # Save the model and vectorizer for our Streamlit dashboard app
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("💾 Model and Vectorizer saved successfully as .pkl files!")