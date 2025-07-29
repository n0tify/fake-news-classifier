import pandas as pd
import os

true_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/True.csv"))
fake_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/Fake.csv"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/news.csv"))

if not os.path.exists(true_path) or not os.path.exists(fake_path):
    print("❌ ERROR: True.csv or Fake.csv not found in data/.")
    exit(1)

# Load both CSVs
true_df = pd.read_csv(true_path)
fake_df = pd.read_csv(fake_path)

# Add labels
true_df["label"] = "REAL"
fake_df["label"] = "FAKE"

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df[["title", "text", "label"]]
df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final news.csv
df.to_csv(output_path, index=False)
print(f"✅ Merged dataset saved to {output_path}")
