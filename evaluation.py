import os

import pandas as pd
import json

from langchain_community.vectorstores import FAISS




GROUND_TRUTH_FILENAME = "ground_truth.json"
queries = [
    {"name": "Italian in Las Vegas", "category": "Italian", "city": "Las Vegas"},
    {"name": "Chinese 5 stars", "category": "Chinese", "stars": 5},
    {"name": "Mexican in Phoenix", "category": "Mexican", "city": "Phoenix"},
    {"name": "Japanese low rating", "category": "Japanese", "stars": 2},
    {"name": "Thai food", "category": "Thai"}
]

def load_ground_truth():
    if os.path.exists(GROUND_TRUTH_FILENAME):
        print("Loading existing ground truth...")

        with open(GROUND_TRUTH_FILENAME, "r") as f:
            gt = json.load(f)
        # 转回 set（方便后面计算）
        gt = {k: set(v) for k, v in gt.items()}

        return gt

    print("Ground truth not found. Building...")
    # 1. load csv
    df = pd.read_csv("data/all_reviews_dataset.csv")

    print("Dataset loaded:", df.shape)
    print("Columns:", df.columns)

    # 2. build ground truth
    ground_truth = {}

    for q in queries:
        df_filtered = df.copy()

        if "category" in q:
            df_filtered = df_filtered[
                df_filtered["categories"].str.contains(q["category"], na=False)
            ]

        if "city" in q and "city" in df.columns:
            df_filtered = df_filtered[df_filtered["city"] == q["city"]]

        if "stars" in q:
            df_filtered = df_filtered[df_filtered["review_stars"] == q["stars"]]

        ids = set(df_filtered["review_id"])

        ground_truth[q["name"]] = ids

        print(f"{q['name']}: {len(ids)} results")

    # 3. store result
    gt_save = {k: list(v) for k, v in ground_truth.items()}

    with open(GROUND_TRUTH_FILENAME, "w") as f:
        json.dump(gt_save, f, indent=2)

    print("Ground truth saved!")
    return ground_truth


def evaluate_model(vs: FAISS):
    print("\n===== Running Evaluation =====\n")

    ground_truth = load_ground_truth()
    results = []

    for query_name, gt_ids in ground_truth.items():

        # 1. retrieval（query）
        docs = vs.similarity_search(query_name, k=50)

        # 2. get review_id
        retrieved_ids = set(
            doc.metadata["review_id"]
            for doc in docs
            if "review_id" in doc.metadata
        )

        # 3. calculate
        correct = gt_ids & retrieved_ids

        precision = len(correct) / len(retrieved_ids) if retrieved_ids else 0
        recall = len(correct) / len(gt_ids) if gt_ids else 0

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        results.append((precision, recall, f1))

        print(f"{query_name}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("-" * 40)

    # avg result
    avg_p = sum(r[0] for r in results) / len(results)
    avg_r = sum(r[1] for r in results) / len(results)
    avg_f1 = sum(r[2] for r in results) / len(results)

    print("\n===== Average Performance =====")
    print(f"Avg Precision: {avg_p:.4f}")
    print(f"Avg Recall:    {avg_r:.4f}")
    print(f"Avg F1 Score:  {avg_f1:.4f}")

    return avg_p, avg_r, avg_f1