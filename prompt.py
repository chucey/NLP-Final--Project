import os
import re
from dotenv import load_dotenv
import rag_retrival
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, device: str = None) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Loads a language model and its tokenizer from Hugging Face."""
    
    print(f"Loading model: {model_name} ...")

    if device is None:
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = str(device).strip().lower()
        aliases = {
            "gpu": "cuda",
            "cuda": "cuda",
            "cpu": "cpu",
            "mps": "mps",
        }
        if resolved_device not in aliases:
            raise ValueError("device must be one of: CPU, CUDA/GPU, MPS")
        resolved_device = aliases[resolved_device]

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model_kwargs = {"torch_dtype": "auto"}
    if resolved_device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if resolved_device != "cuda":
        model = model.to(torch.device(resolved_device))

    dtype = next(model.parameters()).dtype
    model.eval()

    print(f"Model loaded on {resolved_device.upper()} (dtype={dtype})")
    return tok, model


# ---------------------------------------------------------------------------
#  Prompt definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a professional analyst that summarizes Yelp reviews. You MUST follow these rules strictly:

RULES:
1. ONLY use information found in the reviews provided by the user. Do NOT invent, fabricate, or assume any details.
2. If the reviews do not contain enough information for a section, explicitly state "Not enough data."
3. Representative quotes MUST be taken from the reviews in the |Content| section.. Do NOT paraphrase or create fake quotes.
4. Do NOT hallucinate business names, reviewer names, menu items, or any other details not present in the reviews.
5. If no reviews are provided, respond ONLY with: "No reviews found for the given criteria."
6. Do NOT write in first person ("I", "we"). Write as a neutral third-party analyst.
7. List exactly 3 items per bullet-point section. If fewer than 3 exist, write "Not enough data" for the remaining items.

DO NOT:
- Invent menu items or dish names that are not explicitly mentioned in the reviews.
- Fabricate star ratings or percentages that are not stated in the reviews.
- Use phrases like "many reviewers said" unless at least 3 reviews support the claim.
- Add information from your own knowledge. Rely ONLY on the provided reviews.

OUTPUT FORMAT (you must follow this structure exactly):

## Overall Sentiment
[positive / mixed / negative] — one sentence explanation referencing the review statistics

## Top Praised Theme - include business name when providing evidence
- [theme 1]: brief explanation with evidence from reviews
- [theme 2]: brief explanation with evidence from reviews
- [theme 3]: brief explanation with evidence from reviews

## Top Complaints
- [complaint 1]: brief explanation with evidence from reviews - include business name and location when providing evidence
- [complaint 2]: brief explanation with evidence from reviews
- [complaint 3]: brief explanation with evidence from reviews

## Representative Quotes - take verbatim from Content: section of each review
1. "[exact verbatim quote from a review]" — (Business: [name], Stars: [N])
2. "[exact verbatim quote from a review]" — (Business: [name], Stars: [N])
3. "[exact verbatim quote from a review]" — (Business: [name], Stars: [N])

## Confidence Level
[low / medium / high] — based on the number and consistency of reviews analyzed\
"""


def _extract_review_stats(docs: str) -> dict:
    """Extract metadata statistics from the formatted review string.
    
    Parses the output format of retrieve_reviews_for_summary() to compute
    review count, average stars, and unique business count. These stats
    are injected into the user prompt so the LLM has better context for
    generating accurate Overall Sentiment and Confidence Level.
    """
    star_values = [float(m) for m in re.findall(r"Review Stars:\s*([\d.]+)", docs)]
    business_names = set(re.findall(r"Business Name:\s*([^|]+)", docs))
    business_names = {n.strip() for n in business_names if n.strip() and n.strip().lower() != "none"}

    return {
        "review_count": len(star_values) if star_values else docs.count("---"),
        "avg_stars": round(sum(star_values) / len(star_values), 1) if star_values else None,
        "business_count": len(business_names),
        "business_names": sorted(business_names)[:10],  # cap at 10 for prompt brevity
    }


def summarize_reviews(docs: str, tok: AutoTokenizer, model: AutoModelForCausalLM) -> str:
    """Generates a structured summary of Yelp reviews using the loaded LLM.
    
    Args:
        docs (str): A formatted string containing retrieved reviews from the RAG system.
                     This is the output of retrieve_reviews_for_summary().
        tok (AutoTokenizer): The tokenizer for the loaded model.
        model (AutoModelForCausalLM): The loaded language model.

    Returns:
        str: A structured summary of the reviews, or a "no reviews found" message if docs is empty.
    """

    # --- Handle empty RAG results ---
    if not docs or not docs.strip():
        return "No reviews found for the given criteria."

    # Also catch the RAG "no results" sentinel values
    if docs.strip().lower() in ("no results found.", "no results found"):
        return "No reviews found for the given criteria."

    # --- Extract metadata statistics for richer context ---
    stats = _extract_review_stats(docs)

    stats_block = f"""REVIEW STATISTICS (use these to inform your Overall Sentiment and Confidence Level):
- Total reviews analyzed: {stats['review_count']}
- Average star rating: {stats['avg_stars'] if stats['avg_stars'] else 'unknown'}
- Unique businesses covered: {stats['business_count']}"""

    if stats["business_names"]:
        stats_block += f"\n- Business names: {', '.join(stats['business_names'])}"

    # --- Build user prompt with the reviews ---
    user_prompt = f"""{stats_block}

Please summarize the following Yelp reviews:

{docs}"""

    # --- Use the tokenizer's built-in chat template (ChatML for Qwen3) ---
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    conversation = tok.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False,  # disable thinking mode for direct output
    )

    inputs = tok(conversation, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1536,
            min_new_tokens=100,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.2,
            eos_token_id=tok.eos_token_id,
        )
        # Decode only the new tokens generated by the model
        response = tok.decode(
            outputs[0, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
    return response


if __name__ == "__main__":

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    print("=====Loading vectorstore...=====")
    rag = rag_retrival.load_vectorstore()
    print("=====Vectorstore loaded.=====")

    print("=====Retrieving reviews...=====")
    metadata_filter = {
        "categories": "Italian",
        'business_name': None,
        "city": None,
        "state": None,
        "review_stars": None
    }
    docs = rag_retrival.retrieve_reviews_for_summary(rag, metadata_filter=metadata_filter, k=10)
    print("=====Reviews retrieved.=====")

    if torch.backends.mps.is_available():
        device = "MPS"
        print("Running on Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "CUDA"
        print("Running on NVIDIA GPU (CUDA)")
    else:
        device = "CPU"
        print("Running on CPU")

    print("=====Loading language model...=====")
    tok, model = load_model(MODEL_NAME, device=device)

    print("=====Generating summary...=====")
    summary = summarize_reviews(docs, tok, model)
    print("=====Summary generated.=====\n")
    print("=====BUSINESS SUMMARY=====")
    print(summary)
