import streamlit as st
import joblib
import os

st.title("Sentiment Analysis Debugging")

# Check if files exist
if not os.path.exists("sentiment_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("Error: Model or vectorizer file missing! Retrain and save them again.")
    st.stop()

# Load model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    st.write(f"Model Type: {type(model)}")  # Should NOT be <class 'str'>
    st.write(f"Vectorizer Type: {type(vectorizer)}")

    if isinstance(model, str):
        st.error("Error: Loaded model is a string! Please retrain and save it correctly.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()
