'''
This file evaluates the RAG retrieval system by loading the RAG and ground truth labels, performing retrieval based on the same query used for generating the ground truth summaries, and comparing the retrieved documents against the ground truth labels. The evaluation focuses on measuring the overlap between the retrieved review IDs and the ground truth review IDs to assess the effectiveness of the retrieval process in capturing relevant information for summarization using different embeddings and chunk sizes. This code is intended to be run after the RAG system has been built and the ground truth labels have been generated, and it assumes that both the RAG index and the ground truth labels are available in the appropriate format.
'''
from langchain_community.vectorstores import FAISS
import pandas as pd
import build_rag
import rag_retrival
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_ground_truth_labels(labels_path: str = "data/ground_truth_labels.pkl") -> list[dict]:
    """
    Loads the ground truth labels from a pickle file.

    Args:
        labels_path (str, optional): The path to the pickle file containing the ground truth labels. Defaults to "ground_truth_labels.pkl".

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains a query, a metadata filter, and a set of relevant review IDs.

        exmple output: [
            {'query': 'breakfast spots in Alton',
            'metadata_filter': {'city': 'Alton', 'categories': 'Breakfast'},
            'review_ids': {'lkqx0KRpwch6oH2N91-BCw','pIGUls8vJpPMd5iviM-5ng',Sy8THMlpspwuQdauA'}}, ...
                        ]
    """
    with open(labels_path, "rb") as f:
        ground_truth_labels = pickle.load(f)
    return ground_truth_labels

def evaluate_retrieval(vs: FAISS, ground_truth_labels: list[dict], use_metadata_filter: bool = True) -> tuple[float, float, float]:
    """
    Evaluates the retrieval performance of the RAG system by comparing the retrieved review IDs against the ground truth review IDs for each query.

    Args:
        vs (FAISS): The loaded RAG system (vectorstore).
        ground_truth_labels (list[dict]): A list of dictionaries containing queries, metadata filters, and sets of relevant review IDs.
        use_metadata_filter (bool, optional): Whether to use metadata filters for retrieval. Defaults to True.

    Returns:
        tuple[float, float, float]: A tuple containing the average precision, recall, and F1 score across all queries.
    """
    
    precisions = []
    recalls = []
    f1_scores = []

    for label in ground_truth_labels:
        query = label["query"]
        metadata_filter = label["metadata_filter"] if use_metadata_filter else None
        relevant_review_ids = label["review_ids"]

        if use_metadata_filter and metadata_filter:
            retrieved_review_ids = rag_retrival.retrieve_reviews_for_summary(
                vs,
                metadata_filter, # unpack metadata filter for retrieval
                query=query,
                eval_mode=True, # return retrieved review IDs for evaluation
                k=len(relevant_review_ids) * 2  # Retrieve more to account for potential noise in retrieval
        )
        else:
            retrieved_review_ids = rag_retrival.retrieve_reviews_for_summary(
                vs,
                query=query,
                eval_mode=True, # return retrieved review IDs for evaluation
                k=len(relevant_review_ids) * 2  # Retrieve more to account for potential noise in retrieval
            )

        num_relevant = len(relevant_review_ids)
        num_retrieved = len(retrieved_review_ids)
        num_correctly_retrieved = len(set(retrieved_review_ids) & set(relevant_review_ids))

        precision = num_correctly_retrieved / num_retrieved if num_retrieved > 0 else 0.0
        recall = num_correctly_retrieved / num_relevant if num_relevant > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1_score = sum(f1_scores) / len(f1_scores)

    return avg_precision, avg_recall, avg_f1_score

def plot_evaluation_results(results_df: pd.DataFrame):
    # Use your in-memory results dataframe
    plot_df = results_df.copy()

    # Make labels cleaner
    model_short = {
        "BAAI/bge-base-en-v1.5": "bge-base",
        "BAAI/bge-small-en-v1.5": "bge-small",
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2": "mpnet-base-v2",
    }
    plot_df["embedding_model_short"] = plot_df["embedding_model"].map(model_short).fillna(plot_df["embedding_model"])
    plot_df["metadata_flag"] = plot_df["use_metadata_filter"].map({True: "Metadata=True", False: "Metadata=False"})

    # Long format for grouped bars
    long_df = plot_df.melt(
        id_vars=["embedding_model_short", "chunk_size", "metadata_flag"],
        value_vars=["avg_precision", "avg_recall", "avg_f1_score"],
        var_name="metric",
        value_name="score"
    )

    metric_labels = {
        "avg_precision": "Precision",
        "avg_recall": "Recall",
        "avg_f1_score": "F1"
    }
    long_df["metric"] = long_df["metric"].map(metric_labels)

    # Keep deterministic ordering
    row_order = sorted(long_df["chunk_size"].unique().tolist())
    col_order = sorted(long_df["embedding_model_short"].unique().tolist())

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=long_df,
        kind="bar",
        x="metric",
        y="score",
        hue="metadata_flag",
        row="chunk_size",
        col="embedding_model_short",
        row_order=row_order,
        col_order=col_order,
        height=2.4,
        aspect=1.15,
        palette=["#4C78A8", "#F58518"],
        legend=True,
        sharey=True
    )

    g.set_axis_labels("", "Score")
    g.set_titles(row_template="chunk_size={row_name}", col_template="{col_name}")

    # Make bars easier to read
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)

    g.fig.subplots_adjust(top=0.92, hspace=0.32, wspace=0.15)
    g.fig.suptitle("RAG Retrieval Metrics by Embedding Model, Chunk Size, and Metadata Filter", fontsize=14)

    plt.show()
    # save the plot
    plot_save_path = "data/retrieval_evaluation_plot.png"
    g.savefig(plot_save_path)
    print(f"Saved evaluation plot to {plot_save_path}")
   
            
if __name__ == "__main__":
    # define paths
    print("Defining paths...")
    all_reviews_csv_path = "data/all_reviews_dataset.csv"
    eval_results_path = "data/retrieval_evaluation_results.csv"
    labels_path = "data/ground_truth_labels.pkl"
    eval_rag_dir = "eval_rags"  # This should point to the directory containing the FAISS index built with the test parameters

    # define embeddings and chunk sizes to evaluate
    print("Defining evaluation parameters...")
    embedding_models = ["sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                        "BAAI/bge-small-en-v1.5",
                        "BAAI/bge-base-en-v1.5"
                        ]
    
    chunk_sizes = [800, 289, 137, 81]
    use_metadata_filter = True  # Set to False to evaluate retrieval without metadata filtering
    # load ground truth labels
    print("Loading ground truth labels...")
    ground_truth_labels = load_ground_truth_labels(labels_path)

    # evaluate retrieval for each combination of embedding model and chunk size
    print("Evaluating retrieval performance...")
    results = []
    for model_name in embedding_models:
        for chunk_size in chunk_sizes:
            print(f"Evaluating for model: {model_name}, chunk size: {chunk_size}...")
            # build rag index with the specified parameters (if not already built)
            # Note: In practice, you would want to build the RAG index separately and ensure that the eval_rag_dir contains the correct index for the current parameters before running this evaluation code.
            try:
                index_dir = os.path.join(eval_rag_dir, f"{model_name.replace('/', '_')}_chunk{chunk_size}")
                if not os.path.exists(index_dir):
                    print(f"Building RAG index for model: {model_name}, chunk size: {chunk_size}...")
                    build_rag.build_index(
                        csv_path=all_reviews_csv_path,
                        model_name=model_name,
                        chunk_size=chunk_size,
                        index_dir=index_dir
                    )
                    print(f"Built RAG index for model: {model_name}, chunk size: {chunk_size}.")
                else:
                    print(f"RAG index for model: {model_name}, chunk size: {chunk_size} already exists. Skipping build.")
                    
                # load the corresponding RAG index
                print(f"Loading RAG index for model: {model_name}, chunk size: {chunk_size}...")
                vs = rag_retrival.load_vectorstore(index_dir=index_dir, model=model_name)

                # evaluate retrieval performance
                # evaluate retreive with both metadata filtering and wothout metadata filtering to see the difference in performance    
                for use_metadata in [True, False]:
                    print(f"Evaluating retrieval with metadata filter: {use_metadata}...")
                    avg_precision, avg_recall, avg_f1_score = evaluate_retrieval(vs, ground_truth_labels, use_metadata_filter=use_metadata)
                    results.append({
                        "embedding_model": model_name,
                        "chunk_size": chunk_size,
                        "use_metadata_filter": use_metadata,
                        "avg_precision": avg_precision,
                        "avg_recall": avg_recall,
                        "avg_f1_score": avg_f1_score
                    })
                    print(f"Evaluation results for model: {model_name}, chunk size: {chunk_size}, metadata filter: {use_metadata} - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1_score:.4f}")
            except Exception as e:
                print(f"[ERROR] Error evaluating model: {model_name}, chunk size: {chunk_size} - {e}")
                continue

    # save evaluation results to CSV
    print(f"Saving evaluation results to {eval_results_path}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_results_path, index=False)
    print("Saved evaluation results.")
    plot_evaluation_results(results_df)




