import streamlit as st
import pickle
import pandas as pd
import numpy as np
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# configure the streamlit page
st.set_page_config(
    page_title="🧠 NewsVerdict",
    page_icon="📰",
    layout="centered"
)

# apply custom css styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1d1e33, #3e1f47, #461f61, #3f2b96, #5f2c82) !important;
    background-size: 400% 400%;
    animation: galaxyShift 20s ease infinite;
    color: white !important;
}

@keyframes galaxyShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

textarea, .stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid #aaa !important;
    border-radius: 10px !important;
}

.stButton button {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}

.stButton button:hover {
    background: linear-gradient(to right, #ff6a00, #ee0979);
    transform: scale(1.05);
}

.result-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 15px;
    font-size: 1.4rem;
    font-weight: bold;
    text-align: center;
    color: white;
    background: linear-gradient(to right, #00c9ff, #92fe9d);
    box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    animation: popIn 0.5s ease-in-out;
}

.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 0.3rem;
    color: #ffffff;
    animation: fadeIn 1s ease-out;
}
.sub-title {
    text-align: center;
    font-size: 1.2rem;
    color: #dddddd;
    animation: fadeIn 1.5s ease-in;
    margin-bottom: 2rem;
}

footer {
    text-align: center;
    font-size: 0.9rem;
    margin-top: 4rem;
    opacity: 0.8;
    color: #cccccc;
}

@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity: 1; transform: translateY(0);}
}
@keyframes popIn {
    0% {opacity: 0; transform: scale(0.95);}
    100% {opacity: 1; transform: scale(1);}
}
</style>
""", unsafe_allow_html=True)

# display the title and subtitle
st.markdown("## 🧠 NewsVerdict ")
st.markdown(
    """
    Verify Before You Trust :
    Curious whether a news piece is genuine or misleading?  
    Drop your news snippet below and let **NewsVerdict** analyze its authenticity with precision and intelligence. 
    """
)
st.markdown("####  Paste the news content you'd like to evaluate:")

# define model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'news_classifier.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl')

# load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as m:
            model = pickle.load(m)
        with open(VECTORIZER_PATH, 'rb') as v:
            vectorizer = pickle.load(v)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ model files not found!")
        st.stop()

model, vectorizer = load_model()

# preprocess user input
@st.cache_data(show_spinner=False)
def preprocess(text):
    stemmer = PorterStemmer()
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# user input box
user_input = st.text_area(
    "🔍Enter news content below:",
    height=200,
    placeholder="example: the government announced major new reforms..."
)

# prediction button
if st.button("🚀 detect now"):
    if not user_input.strip():
        st.warning("⚠️ please enter some text.")
    else:
        with st.spinner("analyzing..."):
            clean_text = preprocess(user_input)
            vectorized = vectorizer.transform([clean_text])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = round(np.max(proba) * 100, 2)

            if prediction == 1:
                label = "🟢 real news"
                bg = "#2ecc71"
            else:
                label = "🔴 fake news"
                bg = "#e74c3c"

            st.markdown(f"""
                <div class="result-box" style="background-color: {bg};">
                    {label}<br>
                    <div style='font-size: 1rem; margin-top: 10px;'>confidence: {confidence}%</div>
                </div>
            """, unsafe_allow_html=True)

# footer
st.markdown("""
<footer>
    <hr>
    built with ❤️ using <b>Streamlit</b>  • 2025
</footer>
""", unsafe_allow_html=True)
