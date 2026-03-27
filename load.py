import pandas as pd
import re

print("Loading data...")

reviews = pd.read_json(
    "yelp_academic_dataset_review.json",
    lines=True,
    nrows=50000
)

business = pd.read_json(
    "yelp_academic_dataset_business.json",
    lines=True
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning review text...")
reviews["clean_text"] = reviews["text"].apply(clean_text)

print("Merging data...")

merged = pd.merge(
    reviews,
    business,
    on="business_id",
    how="inner"
)

merged = merged.rename(columns={
    "stars_x": "review_stars",
    "name": "business_name"
})

df = merged[[
    "business_id",
    "business_name",
    "categories",
    "review_id",
    "clean_text",
    "date",
    "review_stars"
]]

print("\nDataset shape:", df.shape)

print("\nSample data:")
print(df.head())

df.to_csv("all_reviews_dataset.csv", index=False)

print("\nSaved dataset with categories")