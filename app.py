import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('Sentiment_analyse_RFC_model.pkl')

st.title("Sentiment Analysis")

# User Input
text = st.text_area("Enter Text")

if st.button("Predict"):
    prediction = model.predict([text])
    if prediction[0] == 1:
        st.write("Positive")
    else:
        st.write("Negative")