import pandas as pd
import re
# import matplotlib.pyplot as plt
import numpy as np

print("Loading data...")

reviews = pd.read_json(
    "data/yelp_academic_dataset_review.json",
    lines=True,
    nrows=80000
)

print("Review dataset loaded with shape:", reviews.shape)

business = pd.read_json(
    "data/yelp_academic_dataset_business.json",
    lines=True
)

print("Business dataset loaded with shape:", business.shape)

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
    "city",
    "state",
    "categories",
    "review_id",
    "clean_text",
    "date",
    "review_stars"
]]

df["review_length"] = df["clean_text"].apply(lambda x: len(x.split()))

print("\nDataset shape:", df.shape)

print("\nSample data:")
print(df.head(3))

# Some light EDA
# df.head(3)
# df["review_length"].describe()

# df.plot(kind="hist", y="review_length", bins=50, title="Distribution of Review Lengths")
# df.plot(kind="hist", y="review_stars", bins=5, title="Distribution of Review Stars")
# plt.show()

# remove the bottom 10 percent of reviews by length
tenth_percentile = np.percentile(df["review_length"], 10)
print("10th Percentile of Review Lengths:", tenth_percentile)

print("Dropping rows with missing values...")
df.dropna(inplace=True)
df = df[df["review_length"] > tenth_percentile]
print("Dataset info after dropping missing values:")
print(df.info())

# save the cleaned and merged dataset for future use
save_path = "data/all_reviews_dataset.csv"
df.to_csv(save_path, index=False)
print(f"\nSaved dataset with categories to {save_path}")