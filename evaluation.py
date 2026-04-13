import os
import pandas as pd
import json
from langchain_community.vectorstores import FAISS
from build_rag import build_index
from rag_retrival import load_vectorstore
import rag_retrival




GROUND_TRUTH_FILENAME = "ground_truth.json"
queries = [

    # 1. City + Category（基础语义）
    {"name": "Italian restaurants in New Orleans", "categories": "Italian", "city": "New Orleans"},
    {"name": "Chinese restaurants in Philadelphia", "categories": "Chinese", "city": "Philadelphia"},
    {"name": "Mexican restaurants in Edmonton", "categories": "Mexican", "city": "Edmonton"},
    {"name": "Japanese restaurants in Nashville", "categories": "Japanese", "city": "Nashville"},
    {"name": "Thai restaurants in Philadelphia", "categories": "Thai", "city": "Philadelphia"},

    # 2. Category only（纯语义）
    {"name": "Italian restaurants", "categories": "Italian"},
    {"name": "Chinese restaurants", "categories": "Chinese"},
    {"name": "Mexican restaurants", "categories": "Mexican"},
    {"name": "Japanese restaurants", "categories": "Japanese"},
    {"name": "Thai restaurants", "categories": "Thai"},

    #  3. State + Category（中等难度）
    {"name": "Italian restaurants in PA", "categories": "Italian", "state": "PA"},
    {"name": "Chinese restaurants in NV", "categories": "Chinese", "state": "NV"},
    {"name": "Mexican restaurants in AZ", "categories": "Mexican", "state": "AZ"},

    # 4. Exact rating（高 precision 场景）
    {"name": "Chinese restaurants with 5 star reviews", "categories": "Chinese", "review_stars": 5},
    {"name": "Italian restaurants with 4 star reviews", "categories": "Italian", "review_stars": 4},

    # 5. Range filter（难）
    {"name": "Japanese restaurants with low ratings", "categories": "Japanese", "review_stars": {"op": "lt", "value": 3}},
    {"name": "Mexican restaurants with high ratings", "categories": "Mexican", "review_stars": {"op": "gte", "value": 4}},

    # 6. City + Rating（组合难度）
    {"name": "Chinese restaurants in Nashville with 5 stars", "categories": "Chinese", "city": "Nashville", "review_stars": 5},
    {"name": "Italian restaurants in New Orleans with high ratings", "categories": "Italian", "city": "New Orleans", "review_stars": {"op": "gte", "value": 4}},

    #  7. Hard（多条件 ）
    {"name": "Japanese restaurants in PA with low ratings", "categories": "Japanese", "state": "PA", "review_stars": {"op": "lt", "value": 3}},
    {"name": "Thai restaurants in LA with high ratings", "categories": "Thai", "state": "LA", "review_stars": {"op": "gte", "value": 4}},
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
        if use_filter :
            match = next((q for q in queries if q["name"] == query_name), None)

            if match is None:
                print("Warning: skipping " + query_name + ", not in queries. You should update ground truth.")
                continue

            metadata_filter = match.copy()
            metadata_filter.pop("name")



        retrieved_ids = rag_retrival.retrieve_reviews(vs,query_name, metadata_filter, k)

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
                model_name=model_name,
                chunk_size=chunk,
                index_dir=index_dir,
                chunk_overlap = int(0.15 * chunk),
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
            vs = load_vectorstore(index_dir=index_dir, model_name=model_name)
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

if __name__ == "__main__":

    gene_all_models_index()
    results = evaluate_all_models()
    df = pd.DataFrame(results)
    df.to_csv(evaluation_result_file, index=False)

    print(f"\nResults saved to {evaluation_result_file}")