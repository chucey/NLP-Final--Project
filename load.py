import pandas as pd
import re
import numpy as np

print("Loading data...")

df = pd.read_csv("data/all_reviews_dataset.csv")

print("Dataset loaded with shape:", df.shape)
print("Columns:", df.columns.tolist())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

text_col = None
for candidate in ["text", "review_text", "clean_text", "review", "content"]:
    if candidate in df.columns:
        text_col = candidate
        break

if text_col is None:
    raise ValueError(f"No review text column found. Available columns: {df.columns.tolist()}")

print(f"Using review text column: {text_col}")
df["clean_text"] = df[text_col].apply(clean_text)

if "business_name" not in df.columns and "name" in df.columns:
    df = df.rename(columns={"name": "business_name"})

if "review_stars" not in df.columns:
    if "stars" in df.columns:
        df = df.rename(columns={"stars": "review_stars"})
    elif "rating" in df.columns:
        df = df.rename(columns={"rating": "review_stars"})

required_defaults = {
    "business_id": "",
    "business_name": "",
    "city": "",
    "state": "",
    "categories": "",
    "review_id": "",
    "date": "",
    "review_stars": np.nan,
}

for col, default in required_defaults.items():
    if col not in df.columns:
        df[col] = default

df = df[
    [
        "business_id",
        "business_name",
        "city",
        "state",
        "categories",
        "review_id",
        "clean_text",
        "date",
        "review_stars",
    ]
]

df["review_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))

print("\nDataset shape:", df.shape)
print("\nSample data:")
print(df.head(3))
