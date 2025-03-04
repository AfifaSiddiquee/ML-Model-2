import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title of the app
st.title("Sentiment Analysis App")

# Text input for user
user_input = st.text_area("Enter your review:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Preprocess and transform input text
        input_tfidf = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(input_tfidf)

        # Map prediction to sentiment label
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display result
        st.write(f"### Sentiment: {sentiment}")
