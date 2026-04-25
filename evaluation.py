import os
import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from build_rag import build_index
from rag_retrival import load_vectorstore
import rag_retrival
import matplotlib.pyplot as plt
import numpy as np



GROUND_TRUTH_FILENAME = "ground_truth.json"
queries = [

    # 1. City + Category（基础语义）
    {
        "name": "Italian restaurants in New Orleans",
        # query ：nature language
        "query": "customer reviews about Italian restaurants in New Orleans focusing on food quality and service",
        "categories": "Italian",
        "city": "New Orleans"
    },
    {
        "name": "Chinese restaurants in Philadelphia",
        "query": "reviews of Chinese restaurants in Philadelphia including food and service experience",
        "categories": "Chinese",
        "city": "Philadelphia"
    },
    {
        "name": "Mexican restaurants in Edmonton",
        "query": "customer reviews about Mexican restaurants in Edmonton and overall dining experience",
        "categories": "Mexican",
        "city": "Edmonton"
    },
    {
        "name": "Japanese restaurants in Nashville",
        "query": "reviews of Japanese restaurants in Nashville focusing on food quality and service",
        "categories": "Japanese",
        "city": "Nashville"
    },
    {
        "name": "Thai restaurants in Philadelphia",
        "query": "customer reviews about Thai restaurants in Philadelphia and customer experience",
        "categories": "Thai",
        "city": "Philadelphia"
    },

    # 2. Category only（纯语义）
    {
        "name": "Italian restaurants",
        "query": "customer reviews about Italian restaurants including food quality and service",
        "categories": "Italian"
    },
    {
        "name": "Chinese restaurants",
        "query": "reviews about Chinese restaurants and dining experience",
        "categories": "Chinese"
    },
    {
        "name": "Mexican restaurants",
        "query": "customer reviews of Mexican restaurants focusing on food and service",
        "categories": "Mexican"
    },
    {
        "name": "Japanese restaurants",
        "query": "reviews about Japanese restaurants including food quality and service experience",
        "categories": "Japanese"
    },
    {
        "name": "Thai restaurants",
        "query": "customer reviews about Thai restaurants and overall dining experience",
        "categories": "Thai"
    },

    # 3. State + Category
    {
        "name": "Italian restaurants in PA",
        "query": "customer reviews about Italian restaurants in Pennsylvania focusing on food and service",
        "categories": "Italian",
        "state": "PA"
    },
    {
        "name": "Chinese restaurants in NV",
        "query": "reviews of Chinese restaurants in Nevada including customer experience",
        "categories": "Chinese",
        "state": "NV"
    },
    {
        "name": "Mexican restaurants in AZ",
        "query": "customer reviews about Mexican restaurants in Arizona and food quality",
        "categories": "Mexican",
        "state": "AZ"
    },

    # 4. Exact rating
    {
        "name": "Chinese restaurants with 5 star reviews",
        "query": "customer reviews of highly rated Chinese restaurants with 5 star ratings",
        "categories": "Chinese",
        "review_stars": 5
    },
    {
        "name": "Italian restaurants with 4 star reviews",
        "query": "reviews of Italian restaurants with 4 star ratings and good customer experience",
        "categories": "Italian",
        "review_stars": 4
    },

    # 5. Range filter
    {
        "name": "Japanese restaurants with low ratings",
        "query": "customer reviews of poorly rated Japanese restaurants with low ratings below 3 stars",
        "categories": "Japanese",
        "review_stars": {"op": "lt", "value": 3}
    },
    {
        "name": "Mexican restaurants with high ratings",
        "query": "customer reviews of highly rated Mexican restaurants with ratings above 4 stars",
        "categories": "Mexican",
        "review_stars": {"op": "gte", "value": 4}
    },

    # 6. City + Rating
    {
        "name": "Chinese restaurants in Nashville with 5 stars",
        "query": "customer reviews of 5 star Chinese restaurants in Nashville with great food and service",
        "categories": "Chinese",
        "city": "Nashville",
        "review_stars": 5
    },
    {
        "name": "Italian restaurants in New Orleans with high ratings",
        "query": "customer reviews of highly rated Italian restaurants in New Orleans with great dining experience",
        "categories": "Italian",
        "city": "New Orleans",
        "review_stars": {"op": "gte", "value": 4}
    },

    # 7. Hard（多条件）
    {
        "name": "Japanese restaurants in PA with low ratings",
        "query": "customer reviews of poorly rated Japanese restaurants in Pennsylvania with ratings below 3 stars",
        "categories": "Japanese",
        "state": "PA",
        "review_stars": {"op": "lt", "value": 3}
    },
    {
        "name": "Thai restaurants in LA with high ratings",
        "query": "customer reviews of highly rated Thai restaurants in LA (Louisiana) with ratings above 4 stars",
        "categories": "Thai",
        "state": "LA",
        "review_stars": {"op": "gte", "value": 4}
    },
]

evaluation_result_file = "evaluation_results.csv"

def apply_metadata_filter_df(df, metadata_filter):
    df_filtered = df.copy()

    for k, v in metadata_filter.items():

        if k not in df_filtered.columns:
            continue

        col = df_filtered[k]

        # ===== 情况1：operator（最重要）=====
        if isinstance(v, dict):
            op = v.get("op")
            val = v.get("value")

            # 强制转 numeric（防止 string 类型）
            col_numeric = pd.to_numeric(col, errors="coerce")

            if op == "lt":
                df_filtered = df_filtered[col_numeric < val]
            elif op == "lte":
                df_filtered = df_filtered[col_numeric <= val]
            elif op == "gt":
                df_filtered = df_filtered[col_numeric > val]
            elif op == "gte":
                df_filtered = df_filtered[col_numeric >= val]
            elif op == "eq":
                df_filtered = df_filtered[col_numeric == val]

        # ===== 情况2：字符串 =====
        elif isinstance(v, str):

            if k == "categories":
                df_filtered = df_filtered[
                    col.astype(str)
                       .str.strip()
                       .str.lower()
                       .str.contains(v.strip().lower(), na=False)
                ]
            else:
                df_filtered = df_filtered[
                    col.astype(str)
                       .str.strip()
                       .str.lower()
                       == v.strip().lower()
                ]

        # ===== 情况3：普通数值 =====
        else:
            # 同样做安全转换
            col_numeric = pd.to_numeric(col, errors="coerce")
            df_filtered = df_filtered[col_numeric == v]

    return df_filtered

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
        metadata = q.copy()
        metadata.pop("name")
        metadata.pop("query")
        df_filtered = df.copy()
        df_filtered = apply_metadata_filter_df(df_filtered, metadata)
        ids = set(df_filtered["review_id"])

        ground_truth[q["name"]] = ids

        print(f"{q['name']}: {len(ids)} results")

    # 3. store result
    gt_save = {k: list(v) for k, v in ground_truth.items()}

    with open(GROUND_TRUTH_FILENAME, "w") as f:
        json.dump(gt_save, f, indent=2)

    print("Ground truth saved!")
    return ground_truth


def evaluate_model(vs: FAISS, k: int = 50 , use_filter: bool = False , ground_truth=None ):
    print("\n===== Running Evaluation =====\n")


    results = []

    for query_name, gt_ids in ground_truth.items():

        # 1. retrieval（query）
        metadata_filter = {}
        match = next((q for q in queries if q["name"] == query_name), None)

        if match is None:
            print("Warning: skipping " + query_name + ", not in queries. You should update ground truth.")
            continue
        query = match["query"]
        if use_filter :
            metadata_filter = match.copy()
            metadata_filter.pop("name")
            metadata_filter.pop("query")



        retrieved_ids = rag_retrival.retrieve_reviews_for_summary(vs = vs, metadata_filter= metadata_filter,query= query, k = k ,eval_mode = True)

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

chunk_sizes = [800, 289, 137, 81]
doc_nums = [50, 200]
models_info = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "index_dir_prefix": "minilm",
        "description": "Lightweight baseline model, fast but moderate accuracy"
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "index_dir_prefix": "bge_small",
        "description": "Modern embedding model with better retrieval performance"
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "index_dir_prefix": "mpnet",
        "description": "Stronger transformer-based embedding, high accuracy"
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "index_dir_prefix": "bge_base",
        "description": "Larger BGE model, higher quality but slower"
    },
]


# build all model index
def gene_all_models_index():
    for model in models_info:
        model_name = model["name"]
        prefix = model["index_dir_prefix"]

        for chunk in chunk_sizes:
            index_dir = f"{prefix}_{chunk}"

            print(f"\nBuilding index: {model_name}, chunk={chunk}")

            build_index(
                csv_path= "data/all_reviews_dataset.csv",
                model_name=model_name,
                chunk_size=chunk,
                index_dir=index_dir,
            )


def evaluate_all_models():
    ground_truth = load_ground_truth()
    results = []
    for model in models_info:
        model_name = model["name"]
        prefix = model["index_dir_prefix"]
        describe = model["description"]

        for chunk in chunk_sizes:
            index_dir = f"{prefix}_{chunk}"
            print(f"\nEvaluating: {model_name}, chunk={chunk}")
            vs = load_vectorstore(index_dir=index_dir, model=model_name)
            for num in doc_nums:
                p, r, f1 = evaluate_model(vs=vs,k=num, use_filter=True, ground_truth= ground_truth)

                results.append({
                    "name": model_name,
                    "index_dir": index_dir,
                    "prefix_name": prefix,
                    "use_filter": True,
                    "chunk": chunk,
                    "doc_num": num,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "description" : describe,
                })
                p, r, f1 = evaluate_model(vs=vs,k=num, use_filter=False, ground_truth= ground_truth)

                results.append({
                    "name": model_name,
                    "index_dir": index_dir,
                    "prefix_name": prefix,
                    "use_filter": False,
                    "chunk": chunk,
                    "doc_num": num,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "description" : describe,
                })

    # output
    print("\n===== FINAL RESULTS =====")
    for row in results:
        print(row)

    return results

def plot_summary_bar(df):

    os.makedirs("zARCHIVE", exist_ok=True)

    summary = df.groupby("use_filter")[["precision", "recall", "f1"]].mean()

    if True not in summary.index or False not in summary.index:

        print("Need both metadata modes to plot comparison.")

        return

    meta_true = summary.loc[True]

    meta_false = summary.loc[False]

    labels = ["Precision", "Recall", "F1-Score"]

    true_vals = [meta_true["precision"], meta_true["recall"], meta_true["f1"]]

    false_vals = [meta_false["precision"], meta_false["recall"], meta_false["f1"]]

    x = np.arange(len(labels))

    width = 0.35

    plt.figure(figsize=(8,5))

    plt.bar(x - width/2, true_vals, width, label="Metadata-Aware Retrieval")

    plt.bar(x + width/2, false_vals, width, label="Pure Semantic Retrieval")

    plt.xticks(x, labels)

    plt.ylim(0,1.0)

    plt.ylabel("Score")

    plt.title("Average Retrieval Performance Comparison")

    plt.legend()

    for i,v in enumerate(true_vals):

        plt.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center')

    for i,v in enumerate(false_vals):

        plt.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()

    plt.savefig("zARCHIVE/rag_eval_summary_bar.png", dpi=300)

    plt.show()

def plot_chunk_trend(df):

    os.makedirs("zARCHIVE", exist_ok=True)

    chunk_order = sorted(df["chunk"].unique())

    meta_df = df[df["use_filter"] == True].groupby("chunk")[["recall", "f1"]].mean().loc[chunk_order]
    pure_df = df[df["use_filter"] == False].groupby("chunk")[["recall", "f1"]].mean().loc[chunk_order]

    plt.figure(figsize=(8,5))

    plt.plot(chunk_order, meta_df["recall"], marker='o', label="Metadata Recall")
    plt.plot(chunk_order, meta_df["f1"], marker='o', label="Metadata F1")

    plt.plot(chunk_order, pure_df["recall"], marker='s', linestyle='--', label="Pure Semantic Recall")
    plt.plot(chunk_order, pure_df["f1"], marker='s', linestyle='--', label="Pure Semantic F1")

    plt.xlabel("Chunk Size")
    plt.ylabel("Score")
    plt.title("Chunk Size Impact on Retrieval Performance")
    plt.ylim(0,1.0)
    plt.xticks(chunk_order)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("zARCHIVE/rag_eval_chunk_trend.png", dpi=300)
    plt.show()

if __name__ == "__main__":

    # gene_all_models_index()
    results = evaluate_all_models()
    df = pd.DataFrame(results)
    df.to_csv(evaluation_result_file, index=False)

    plot_summary_bar(df)
    plot_chunk_trend(df)

    print(f"\nResults saved to {evaluation_result_file}")