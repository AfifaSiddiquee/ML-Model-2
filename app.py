import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define sentiment emojis
sentiment_emojis = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😡"
}

# Streamlit App UI
st.set_page_config(page_title="SentimentScope Analyzer", page_icon="🔍")
st.title("📊 SentimentScope Analyzer")
st.write("📝 Curious about what a review really says? Enter any product review, and **SentimentScope Analyzer** will instantly classify its sentiment as **Positive, Neutral, or Negative**—helping you make informed decisions in seconds!")

# Input box for user review
review_text = st.text_area("Enter your review here:")

# Analyze Button
if st.button("Analyze Sentiment"):
    if review_text.strip():
        # Transform the input text
        review_tfidf = vectorizer.transform([review_text])
        
        # Predict sentiment
        sentiment = model.predict(review_tfidf)[0]
        
        # Display result with emoji
        st.markdown(f"**Predicted Sentiment:** {sentiment_emojis[sentiment]} **{sentiment.capitalize()}**")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.markdown("Developed by **Afifa Siddiqui**")
