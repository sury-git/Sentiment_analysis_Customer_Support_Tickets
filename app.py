import streamlit as st
import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('Sentiment_analyse_RFC_model.pkl','rb') as f:
    model = pickle.load(f)

st.title("Sentiment Analysis")

# User Input
text = st.text_area("Enter Text")

if st.button("Predict"):
    prediction = model.predict([text])
    if prediction[0] == 1:
        st.write("Positive")
    else:
        st.write("Negative")