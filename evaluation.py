import pandas as pd
import json

# =====================
# 1. 加载数据
# =====================
df = pd.read_csv("data/all_reviews_dataset.csv")

print("Dataset loaded:", df.shape)
print("Columns:", df.columns)

# =====================
# 2. 定义 queries
# =====================
queries = [
    {"name": "Italian in Las Vegas", "category": "Italian", "city": "Las Vegas"},
    {"name": "Chinese 5 stars", "category": "Chinese", "stars": 5},
    {"name": "Mexican in Phoenix", "category": "Mexican", "city": "Phoenix"},
    {"name": "Japanese low rating", "category": "Japanese", "stars": 2},
    {"name": "Thai food", "category": "Thai"}
]

# =====================
# 3. 构建 ground truth
# =====================
ground_truth = {}

for q in queries:
    df_filtered = df.copy()

    if "category" in q:
        df_filtered = df_filtered[
            df_filtered["categories"].str.contains(q["category"], na=False)
        ]

    if "city" in q:
        if "city" in df.columns:
            df_filtered = df_filtered[df_filtered["city"] == q["city"]]

    if "stars" in q:
        df_filtered = df_filtered[df_filtered["review_stars"] == q["stars"]]

    ids = set(df_filtered["review_id"])

    ground_truth[q["name"]] = ids

    print(f"{q['name']}: {len(ids)} results")

# =====================
# 4. 保存结果
# =====================
gt_save = {k: list(v) for k, v in ground_truth.items()}

with open("ground_truth.json", "w") as f:
    json.dump(gt_save, f, indent=2)

print("Ground truth saved!")