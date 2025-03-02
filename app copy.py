import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import spacy

nlp = spacy.load('en_core_web_sm',disable=['tagger','parser', 'ner']) # disabling Named Entity Recognition for speed
#define a function for text cleaning
def text_cleaner(text):

  #remove user mentions
  text = re.sub(r'@[A-Za-z0-9]+','',text)

  #remove hashtags
  #text = re.sub(r'#[A-Za-z0-9]+','',text)

  #remove links
  text = re.sub(r'http\s+', '', text)

  #convering text to lower case
  text = text.lower()

  # fetch only words
  text = re.sub("[^a-z]+", " ", text)

  # removing extra spaces
  text=re.sub("[\s]+"," ",text)

  # creating doc object
  doc=nlp(text)

  # remove stopwords and lemmatize the text
  tokens=[token.lemma_ for token in doc if(token.is_stop==False)]

  #join tokens by space
  return " ".join(tokens)
# word_vectorizer = TfidfVectorizer(max_features=1000)
with open('word_vectorizer_model.pkl','rb') as f1:
  word_vectorizer = pickle.load(f1)

# Load the trained model
with open('Sentiment_analyse_RFC_model.pkl','rb') as f:
    model = pickle.load(f)

# model = joblib.load('Sentiment_analyse_RFC_model.pkl')

st.title("Sentiment Analysis")

# User Input
text = st.text_area("Enter Text")
clear_text = text_cleaner(text)
vectorizer_text = word_vectorizer.transform(text)
if st.button("Predict"):
    prediction = model.predict([vectorizer_text])
    if prediction[0] == 1:
        st.write("Positive")
    else:
        st.write("Negative")