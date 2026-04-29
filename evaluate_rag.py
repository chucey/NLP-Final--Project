'''
This file evaluates the RAG retrieval system by loading the RAG and ground truth labels, performing retrieval based on the same query used for generating the ground truth summaries, and comparing the retrieved documents against the ground truth labels. The evaluation focuses on measuring the overlap between the retrieved review IDs and the ground truth review IDs to assess the effectiveness of the retrieval process in capturing relevant information for summarization using different embeddings and chunk sizes. This code is intended to be run after the RAG system has been built and the ground truth labels have been generated, and it assumes that both the RAG index and the ground truth labels are available in the appropriate format.

Additionally, this file provides summary evaluation functions (auto_evaluate, llm_judge_evaluate, run_full_evaluation, evaluate_no_result_handling) for evaluating the quality of LLM-generated summaries using both automated metrics and LLM-as-Judge (Gemini).
'''
from langchain_community.vectorstores import FAISS
import pandas as pd
import build_rag
import rag_retrival
import pickle
import os
import re
import json
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

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


# ===========================================================================
#  Summary Evaluation — Automated Metrics + LLM-as-Judge
# ===========================================================================

# Key sections we expect a good summary to contain (based on SYSTEM_PROMPT)
EXPECTED_SECTIONS = [
    "overall sentiment",
    "top praised theme",  # matches SYSTEM_PROMPT header (singular)
    "top complaints",
    "representative quotes",
    "confidence level",
]

SENTIMENT_KEYWORDS = {
    "positive": ["great", "excellent", "amazing", "love", "best", "fantastic", "wonderful", "delicious", "friendly", "perfect"],
    "negative": ["bad", "terrible", "worst", "awful", "horrible", "rude", "disgusting", "cold", "slow", "dirty"],
}


def _check_format_compliance(summary: str) -> dict:
    """Check if the summary follows the expected structured format."""
    summary_lower = summary.lower()
    found = [s for s in EXPECTED_SECTIONS if s in summary_lower]
    missing = [s for s in EXPECTED_SECTIONS if s not in summary_lower]
    score = len(found) / len(EXPECTED_SECTIONS) if EXPECTED_SECTIONS else 0
    return {"score": round(score, 2), "sections_found": found, "sections_missing": missing}


def _check_quote_faithfulness(summary: str, source_reviews: str) -> dict:
    """Check if quoted text in the summary actually appears in the source reviews."""
    quotes = re.findall(r'"([^"]{10,})"', summary)
    quotes += re.findall(r"'([^']{10,})'", summary)
    if not quotes:
        return {"score": 0.5, "total_quotes": 0, "verified_quotes": [], "unverified_quotes": [],
                "note": "No quotes found in summary to verify"}
    source_lower = source_reviews.lower()
    verified, unverified = [], []
    for quote in quotes:
        words = quote.lower().strip().split()
        matched = False
        for ratio in [1.0, 0.75, 0.5]:
            n_words = max(4, int(len(words) * ratio))
            if " ".join(words[:n_words]) in source_lower:
                matched = True
                break
        (verified if matched else unverified).append(quote)
    total = len(quotes)
    return {"score": round(len(verified) / total, 2) if total > 0 else 0, "total_quotes": total,
            "verified_quotes": verified, "unverified_quotes": unverified}


def _check_coverage(summary: str, source_reviews: str) -> dict:
    """Check how well the summary covers themes present in the source reviews."""
    source_lower, summary_lower = source_reviews.lower(), summary.lower()
    themes = ["food", "service", "atmosphere", "price", "wait", "staff", "clean", "portion",
              "flavor", "fresh", "menu", "parking", "reservation", "delivery", "ambiance", "decor", "location"]
    source_themes = [t for t in themes if t in source_lower]
    summary_themes = [t for t in source_themes if t in summary_lower]
    score = len(summary_themes) / len(source_themes) if source_themes else 0
    return {"score": round(score, 2), "themes_in_source": source_themes,
            "themes_in_summary": summary_themes,
            "themes_missed": [t for t in source_themes if t not in summary_themes]}


def _check_hallucination_signals(summary: str, source_reviews: str) -> dict:
    """Check for potential hallucination signals in the summary."""
    flags = []
    for phrase in ["no reviews found", "no reviews available", "i don't know"]:
        if phrase in summary.lower() and source_reviews.strip():
            flags.append(f"Says '{phrase}' but reviews were provided")
    source_businesses = set()
    for match in re.finditer(r"Business Name:\s*([^|]+)", source_reviews):
        name = match.group(1).strip().lower()
        if name and name != "none":
            source_businesses.add(name)
    summary_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', summary)
    for name in summary_names:
        # Skip if the match contains a newline — it's a paragraph opening, not a business name
        if "\n" in name:
            continue
        # Skip "At Xxx Yyy" — prepositional phrases, not business names
        if name.startswith("At "):
            continue
        name_lower = name.lower()
        skip = [
            "overall sentiment", "top praised", "top complaints", "representative quotes",
            "confidence level", "not enough", "yelp reviews", "business name", "review stars",
            # sentiment / markdown heading words that the regex captures by mistake
            "mixed", "positive", "negative", "neutral",
            "final answer", "final output", "detailed analysis",
            "low", "medium", "high", "based on",
            # descriptor words used as bullet sub-headers by 4B model
            "quality", "times", "experience", "preparation", "staffing",
            "items", "customer", "wait", "service", "atmosphere", "issues",
            "slow", "poor", "great", "fresh", "unique", "authentic", "inconsistent",
            "unfriendly", "unpleasant", "creative", "affordable",
        ]
        if any(s in name_lower for s in skip):
            continue
        if source_businesses and not any(name_lower in biz for biz in source_businesses):
            if len(name.split()) >= 2:
                flags.append(f"Possible hallucinated business name: '{name}'")
    return {"score": round(max(0, 1.0 - len(flags) * 0.25), 2), "flags": flags, "num_flags": len(flags)}


def auto_evaluate(summary: str, source_reviews: str) -> dict:
    """Run all automated evaluation metrics on a generated summary.

    Args:
        summary: The generated summary text
        source_reviews: The source reviews string (from RAG retrieval)

    Returns:
        dict with scores for each metric and an overall score
    """
    fmt = _check_format_compliance(summary)
    qt = _check_quote_faithfulness(summary, source_reviews)
    cov = _check_coverage(summary, source_reviews)
    hal = _check_hallucination_signals(summary, source_reviews)
    overall = fmt["score"] * 0.2 + qt["score"] * 0.3 + cov["score"] * 0.25 + hal["score"] * 0.25
    return {"overall_auto_score": round(overall, 2), "format_compliance": fmt,
            "quote_faithfulness": qt, "theme_coverage": cov, "hallucination_check": hal}


# ===========================
# LLM-as-Judge (Gemini)
# ===========================

JUDGE_PROMPT = """\
You are an expert evaluator for text summarization systems. Your task is to evaluate a machine-generated summary of Yelp reviews.

You will be given:
1. SOURCE REVIEWS: The original reviews that were provided to the summarization model
2. GENERATED SUMMARY: The summary produced by the model

Evaluate the summary on the following 5 dimensions. For each, provide a score from 1-5 and a brief justification.

SCORING RUBRIC:

1. **Faithfulness** (Is the summary factually consistent with the source reviews?)
   - 5: Fully faithful, all claims supported by source reviews
   - 1: Completely fabricated / hallucinated

2. **Completeness** (Does the summary capture the main themes?)
   - 5: Covers all major positive and negative themes
   - 1: Does not address the review content at all

3. **Coherence** (Is the summary well-structured and readable?)
   - 5: Perfectly structured, clear, professional
   - 1: Incoherent, unreadable

4. **Relevance** (Does the summary focus on what matters?)
   - 5: Focuses on the most important aspects
   - 1: Completely off-topic

5. **Quote Accuracy** (Are representative quotes actually from the reviews?)
   - 5: All quotes are verbatim from the reviews
   - 1: All quotes are fabricated / no quotes provided

You MUST respond ONLY with a valid JSON object in exactly this format (no markdown, no extra text):
{
    "faithfulness": {"score": <1-5>, "justification": "<brief reason>"},
    "completeness": {"score": <1-5>, "justification": "<brief reason>"},
    "coherence": {"score": <1-5>, "justification": "<brief reason>"},
    "relevance": {"score": <1-5>, "justification": "<brief reason>"},
    "quote_accuracy": {"score": <1-5>, "justification": "<brief reason>"},
    "overall_notes": "<any additional observations>"
}\
"""


def llm_judge_evaluate(summary: str, source_reviews: str, api_key: str,
                       model_name: str = "gemini-2.5-flash") -> dict:
    """Use Gemini as an LLM judge to evaluate the summary.

    Args:
        summary: The generated summary text
        source_reviews: The source reviews string
        api_key: Google AI Studio API key
        model_name: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        dict with scores for each dimension and justifications
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install google-generativeai: pip install google-generativeai")

    genai.configure(api_key=api_key)

    max_source_chars = 30000
    if len(source_reviews) > max_source_chars:
        source_reviews = source_reviews[:max_source_chars] + "\n... [truncated for evaluation]"

    user_message = f"SOURCE REVIEWS:\n{source_reviews}\n\n---\n\nGENERATED SUMMARY:\n{summary}"

    judge_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=JUDGE_PROMPT,
    )

    try:
        response = judge_model.generate_content(
            user_message,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        response_text = response.text.strip()

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)

        dimensions = ["faithfulness", "completeness", "coherence", "relevance", "quote_accuracy"]
        scores = [result[d]["score"] for d in dimensions if d in result]
        result["overall_judge_score"] = round(sum(scores) / len(scores), 2) if scores else 0

        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"Failed to parse judge response as JSON: {e}",
            "raw_response": response_text,
            "overall_judge_score": 0,
        }
    except Exception as e:
        return {
            "error": f"Gemini API error: {e}",
            "overall_judge_score": 0,
        }


def run_full_evaluation(summary: str, source_reviews: str, api_key: Optional[str] = None,
                        gemini_model: str = "gemini-2.5-flash", print_results: bool = True) -> dict:
    """Run both automated and LLM-as-judge evaluation.

    Args:
        summary: The generated summary text
        source_reviews: The source reviews string
        api_key: Google AI Studio API key (optional — skips LLM judge if not provided)
        gemini_model: Gemini model to use for judging
        print_results: Whether to print a formatted report

    Returns:
        dict with all evaluation results
    """
    results = {}

    if print_results:
        print("Running automated evaluation...")
    results["auto"] = auto_evaluate(summary, source_reviews)

    if api_key:
        if print_results:
            print("Running LLM-as-judge evaluation (Gemini)...")
        results["judge"] = llm_judge_evaluate(summary, source_reviews, api_key, gemini_model)
    else:
        results["judge"] = {"skipped": True, "reason": "No API key provided"}

    if print_results:
        print("\n" + "=" * 60)
        print("📊 EVALUATION REPORT")
        print("=" * 60)

        auto = results["auto"]
        print(f"\n🤖 Automated Metrics (Overall: {auto['overall_auto_score']}/1.00)")
        print(f"   Format Compliance:    {auto['format_compliance']['score']}/1.00")
        print(f"   Quote Faithfulness:   {auto['quote_faithfulness']['score']}/1.00")
        print(f"   Theme Coverage:       {auto['theme_coverage']['score']}/1.00")
        print(f"   Hallucination Check:  {auto['hallucination_check']['score']}/1.00")

        if auto["format_compliance"]["sections_missing"]:
            print(f"   ⚠️  Missing sections: {auto['format_compliance']['sections_missing']}")
        if auto["quote_faithfulness"].get("unverified_quotes"):
            print(f"   ⚠️  Unverified quotes: {auto['quote_faithfulness']['unverified_quotes'][:2]}")
        if auto["hallucination_check"]["flags"]:
            print(f"   ⚠️  Flags: {auto['hallucination_check']['flags'][:3]}")

        if "judge" in results and not results["judge"].get("skipped") and not results["judge"].get("error"):
            judge = results["judge"]
            print(f"\n🧑‍⚖️ LLM-as-Judge Scores (Overall: {judge.get('overall_judge_score', 'N/A')}/5.00)")
            for dim in ["faithfulness", "completeness", "coherence", "relevance", "quote_accuracy"]:
                if dim in judge:
                    print(f"   {dim.capitalize():20s} {judge[dim]['score']}/5  — {judge[dim]['justification']}")
            if judge.get("overall_notes"):
                print(f"   📝 Notes: {judge['overall_notes']}")
        elif results["judge"].get("error"):
            print(f"\n❌ LLM Judge Error: {results['judge']['error']}")

        print("\n" + "=" * 60)

    return results


def evaluate_no_result_handling(summarize_fn, tok, model) -> dict:
    """Test that the model correctly handles empty RAG results.

    Args:
        summarize_fn: The summarize_reviews function
        tok: The tokenizer
        model: The language model

    Returns:
        dict with pass/fail and the actual output
    """
    test_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("\n\n", "Newlines only"),
        ("No results found.", "RAG sentinel value"),
    ]

    results = []
    for docs_input, description in test_cases:
        output = summarize_fn(docs_input, tok, model)
        expected_phrases = ["no reviews found", "no reviews available", "i don't know"]
        passed = any(phrase in output.lower() for phrase in expected_phrases)

        results.append({
            "test": description,
            "passed": passed,
            "output": output[:200],
        })
        print(f"  {'✅' if passed else '❌'} {description}: {output[:100]}")

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "results": results}


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
