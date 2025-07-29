import pandas as pd
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "news.csv")
model_dir = os.path.join(BASE_DIR, "model")
vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
model_path = os.path.join(model_dir, "news_classifier.pkl")

# Auto-skip training if already done
if os.path.exists(vectorizer_path) and os.path.exists(model_path):
    print("✅ Model and vectorizer already exist. Skipping training.")
    sys.exit()

# Read CSV
try:
    df = pd.read_csv(data_path, quoting=1, encoding="utf-8", on_bad_lines='skip')
except Exception as e:
    print("❌ Failed to read CSV:", e)
    sys.exit()

# Print column names for debugging
print(f"🧾 Columns found in CSV: {list(df.columns)}")

# Handle column variations
if "text" in df.columns and ("label" in df.columns or "subject" in df.columns):
    label_col = "label" if "label" in df.columns else "subject"
    df = df[["text", label_col]].rename(columns={label_col: "label"})
else:
    print("❌ CSV must contain 'text' and 'label' or 'subject' columns.")
    sys.exit()

# Drop rows with missing values
df = df.dropna()

# Ensure label is string type
df["label"] = df["label"].astype(str)

# Train-test split
X_train, _, y_train, _ = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Save model & vectorizer
os.makedirs(model_dir, exist_ok=True)
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer trained and saved successfully.")
