import streamlit as st
import joblib
import os

st.title("Sentiment Analysis App")

# Ensure files exist
if not os.path.exists("sentiment_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("Error: Model or vectorizer file missing! Retrain and save them again.")
    st.stop()

# Load model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# User input
user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        try:
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"### Sentiment: {sentiment}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
