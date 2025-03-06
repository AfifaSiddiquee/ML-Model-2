import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 14px;
            color: grey;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.markdown("<h1 style='text-align: center; color: #333;'>SentimentScope Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>📝 Curious about what a review really says? Enter any product review, and SentimentScope Analyzer will instantly classify its sentiment as Positive, Neutral, or Negative—helping you make informed decisions in seconds!</p>", unsafe_allow_html=True)

# User input
st.subheader("Enter a review to analyze its sentiment")
review = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if review:
        # Transform input text
        review_tfidf = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Display result with emoji
        sentiment_map = {
            "positive": "😊 Positive",
            "neutral": "😐 Neutral",
            "negative": "😞 Negative"
        }
        st.markdown(f"<h3 style='color: #4CAF50;'>Predicted Sentiment: {sentiment_map[prediction]}</h3>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a review to analyze.")

# Footer with LinkedIn & GitHub links
st.markdown(
    """
    <div class='footer'>
        Developed by <b>Afifa Siddiqui</b> | 
        <a href="https://www.linkedin.com/in/afifa-siddiqui" target="_blank">LinkedIn</a> |
        <a href="https://github.com/AfifaSiddiquee" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
