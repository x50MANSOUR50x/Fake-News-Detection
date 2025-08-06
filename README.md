# 📰 Fake News Detection with GloVe Embeddings and Streamlit

This project uses **GloVe word embeddings** and a **Logistic Regression classifier** to detect whether a news article is **real** or **fake**. It includes a **Streamlit web app** to interactively test the model in real-time.

---

## 🚀 Features

- Preprocessing using NLTK: lowercase, stopword removal, stemming
- Embedding news text using **GloVe 100D**
- Trained Logistic Regression model on 45K+ articles
- Evaluation with accuracy and F1-score
- Interactive Streamlit app to classify user-entered news
- Saved `.pkl` model for reproducible inference

---

## 📂 Dataset

- Source: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Classes:  
  - `1` → Real News  
  - `0` → Fake News  
- Columns used: `title`, `text`, `label`

---

## 📊 Model Performance

| Model              | Accuracy | F1 Score |
|-------------------|----------|----------|
| Logistic Regression | 98.93%   | 98.88%   |
| SVM                | 99.52%   | 99.50%   |
| Random Forest      | 99.67%   | 99.66%   |
| GloVe + LR         | ~97.9%   | ~97.8%   |

---

## 🧠 Preprocessing Steps

- Lowercasing
- HTML & punctuation removal
- Tokenization
- Stopword removal (`nltk`)
- Porter Stemming

---

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Download NLTK resources (first-time only)
```bash
import nltk
nltk.download('stopwords')
```
### 4. Download GloVe 100D embeddings
- GloVe 6B Download
- Extract and place glove.6B.100d.txt in Data/
### 5. Run the Streamlit App
```bash
streamlit run app.py
```
### 📁 Project Structure
```bash
fake-news-detection/
├── app.py                      # Streamlit app
├── glove_logistic_model.pkl    # Saved GloVe model
├── Data/
│   └── glove.6B.100d.txt       # GloVe embeddings
├── Fake_news_detection.ipynb   # Training notebook
├── requirements.txt
└── README.md
```
### 🤝 Credits
Developed by Mohammed Ahmed Mansour
Under guidance from Elevvo Internship Program