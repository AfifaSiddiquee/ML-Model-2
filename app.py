import streamlit as st
import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained logistic regression model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to predict sentiment
def predict_sentiment(review):
    review_cleaned = clean_text(review)
    review_vectorized = vectorizer.transform([review_cleaned])
    prediction = model.predict(review_vectorized)[0]
    
    if prediction == 1:
        return "Positive 😊", "green"
    elif prediction == 0:
        return "Neutral 😐", "orange"
    else:
        return "Negative 😞", "red"

# ------------------------------------------
# 🚀 Streamlit UI - SentimentScope Analyzer
# ------------------------------------------
st.set_page_config(page_title="SentimentScope Analyzer", page_icon="🔍", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    body {
        background-color: #F7F9F9;
    }
    .stTextArea textarea {
        background-color: #F7F9F9;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    </style>
    """, unsafe_allow_html=True)

# 📌 Header Section
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>SentimentScope Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #7F8C8D;'>Analyze Product Reviews with Machine Learning</h3>", unsafe_allow_html=True)
st.write("---")

# 📝 Review Input
st.markdown("### Enter a Review to Analyze its Sentiment")
review_text = st.text_area("Enter your review here:", height=150)

# 🔍 Analyze Button
if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        sentiment, color = predict_sentiment(review_text)
        st.markdown(f"<h2 style='color: {color};'>{sentiment}</h2>", unsafe_allow_html=True)

# 📌 Footer Section
st.write("---")
st.markdown("""
    <p style="text-align: center;">Developed by <b>Afifa Siddiqui</b></p>
    <p style="text-align: center;">
        <a href="https://github.com/AfifaSiddiquee" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/afifa-siddiquii/" target="_blank">LinkedIn</a>
    </p>
    """, unsafe_allow_html=True)
