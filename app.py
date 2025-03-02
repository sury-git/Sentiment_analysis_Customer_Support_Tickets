import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("Sentiment_analyse_RFC_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer (if used during training)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("Sentiment Analysis")

# User Input
text = st.text_area("Enter Text")

if st.button("Predict"):
    if text.strip():  # Ensure input is not empty
        text_vectorized = vectorizer.transform([text])  # Transform input text
        prediction = model.predict(text_vectorized)
        
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text.")
