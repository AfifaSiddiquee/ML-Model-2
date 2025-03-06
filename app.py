import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("Sentiment Analysis App")
st.subheader("Enter a review to analyze its sentiment")

# Input text box
user_input = st.text_area("Enter your review here:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Transform user input using the loaded vectorizer
        user_input_tfidf = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = model.predict(user_input_tfidf)

        # Convert numeric prediction to label
        sentiment_label = "Positive" if prediction[0] == 1 else "Negative"

        # Display result
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
    else:
        st.warning("Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.markdown("Developed by **Afifa Siddiqui**")
