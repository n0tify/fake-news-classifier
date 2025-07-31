# 🧠 NewsVerdict — Fake News Classifier

This project classifies news articles as **Real (1)** or **Fake (0)** using Natural Language Processing (NLP) and Machine Learning techniques.

It features:  
- 🐍 Python (Scikit-learn, NLTK, Pandas)  
- 📊 TF-IDF Vectorization  
- 🤖 Logistic Regression & Naive Bayes  
- 💻 Streamlit for interactive web interface  

---

## 🚀 Objectives

- Clean and process raw news data  
- Extract features using **TF-IDF**  
- Train and evaluate ML models  
- Visualize performance with **confusion matrix**  
- Deploy an intelligent **Streamlit web app**  

---

## 🗂️ Project Structure

```
FAKE-NEWS-CLASSIFIER/
├── app/
│   └── app.py                  ← Streamlit web app
├── data/
│   ├── Fake.csv
│   ├── news.csv
│   └── True.csv
├── models/
│   ├── news_classifier.pkl     ← Final ML model
│   ├── tfidf_vectorizer.pkl    ← TF-IDF vectorizer
│   └── lr_model.pkl            ← Logistic Regression model (optional)
├── notebooks/
│   └── fake_news_classifier.ipynb
├── scripts/
│   ├── merge_data.py
│   └── train_model.py
├── venv/
├── .gitignore
├── requirements.txt
└── README.md
```

---
## 🌍 Live Deployment & Monitoring

### 🔗 Web App

The project is live and accessible at:  
👉 **[https://newsverdict.streamlit.app/](https://newsverdict.streamlit.app/)**

- Paste any news article or headline  
- Instantly see if it's **🟢 Real** or **🔴 Fake**  
- Backed by trained ML model with confidence scores

### 📊 Uptime Monitoring

Uptime and performance are monitored via UptimeRobot:  
👉 **[https://dashboard.uptimerobot.com/monitors](https://dashboard.uptimerobot.com/monitors)**

- Live server status  
- Response time tracking  
- Downtime alerts and reliability insights
 ---
## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧪 Training & Evaluation

### 🔹 Dataset

We use a merged dataset from `True.csv` and `Fake.csv`.

### 🔹 Preprocessing

- Lowercasing  
- URL & punctuation removal  
- Lemmatization using `WordNetLemmatizer`  
- Stopword removal  

### 🔹 Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(cleaned_texts)
```

### 🔹 Model Training

```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
```

### 🔹 Evaluation

**Logistic Regression**  
- Accuracy: `~77.7%`  
- Weighted F1 Score: `~76%`  

**Naive Bayes**  
- Accuracy: `~69.5%`  
- Weighted F1 Score: `~65.7%`  

Confusion matrix is also plotted using Seaborn.

---

## 💾 Model Saving

```python
import joblib

joblib.dump(lr_model, "models/news_classifier.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
```

---

## 🌐 Streamlit App

To launch the app locally:

```bash
streamlit run app/app.py
```

The app allows you to paste any news snippet and instantly see if it's **🟢 Real** or **🔴 Fake** — with confidence percentage.

---

## 📈 Visualization

Confusion matrix heatmap:

```python
plt.figure(figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
```

---

## ✅ Deployment Tips

Make sure `models/news_classifier.pkl` and `models/tfidf_vectorizer.pkl` are:  
- Present in the GitHub repo ✅  
- Not excluded by `.gitignore` ❌  

---

## 📌 Notes

- Dataset contains categories like `'News'`, `'politics'`, `'worldnews'`, etc.  
- Supports multi-class classification.  
- Best results from Logistic Regression.  

---

## 🙌 Acknowledgements
 
- [NLTK](https://www.nltk.org/)  
- [Scikit-learn](https://scikit-learn.org/)  
- [Streamlit](https://streamlit.io/)  

---

> Built with ❤️ using Streamlit · 2025
