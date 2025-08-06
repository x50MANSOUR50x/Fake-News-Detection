import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib

# Load GloVe embeddings
@st.cache_resource
def load_glove_embeddings(glove_path, dim=100):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# Convert text to GloVe vector
def text_to_glove_vector(text, embeddings, dim=100):
    words = text.split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# Load model and GloVe
model = joblib.load("glove_logistic_model.pkl")
glove_path = "Data/glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_path)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App (GloVe Model)")
st.write("Enter a news article or headline below to check if it's real or fake.")

user_input = st.text_area("🧾 News Text", height=200)

if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        preprocessed_text = preprocess(user_input)
        vector = text_to_glove_vector(preprocessed_text, glove_embeddings)
        prediction = model.predict(vector.reshape(1, -1))[0]
        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.markdown(f"### Prediction: {label}")
