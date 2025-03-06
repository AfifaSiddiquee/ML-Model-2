import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📊 SentimentScope Analyzer")

# Add your final description
st.markdown(
    """
    **📝 Curious about what a review really says?**  
    Enter any product review, and **SentimentScope Analyzer** will instantly classify its sentiment as  
    **Positive, Neutral, or Negative**—helping you make informed decisions in seconds!  
    """
)

# Text input for user review
review = st.text_area("Enter your review here:", "")

# Prediction button
if st.button("Analyze Sentiment"):
    if review:
        # Preprocess the input review
        review_tfidf = vectorizer.transform([review])
        prediction = model.predict(review_tfidf)[0]
        
        # Display the result
        st.markdown(f"**Predicted Sentiment:** :blue[{prediction}]")
    else:
        st.warning("⚠️ Please enter a review before analyzing.")

# Footer
st.markdown("---")
st.markdown("Developed by **Afifa Siddiqui**")

