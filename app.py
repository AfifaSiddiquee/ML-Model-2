import streamlit as st
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Title and description
st.title("SentimentScope Analyzer")
st.markdown(
    "📝 Curious about what a review really says? Enter any product review, and **SentimentScope Analyzer** "
    "will instantly classify its sentiment as **Positive**, **Neutral**, or **Negative**—helping you make informed decisions in seconds!"
)

# User input field
user_input = st.text_area("✍️ Enter your product review:", "")

# Sentiment Emojis Dictionary
sentiment_emojis = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😞"
}

# Label Encoding Mapping
label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

# Analyze Sentiment Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        # Transform the user input using the TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])

        # Predict sentiment (returns 0, 1, or 2)
        predicted_label = model.predict(input_tfidf)[0]

        # Convert numerical prediction to sentiment text
        sentiment = label_mapping.get(predicted_label, "unknown")

        # Display the sentiment result with emoji
        if sentiment in sentiment_emojis:
            st.markdown(f"**Predicted Sentiment:** {sentiment_emojis[sentiment]} **{sentiment.capitalize()}**")
        else:
            st.error("⚠️ Unexpected sentiment prediction. Please check the model output.")
