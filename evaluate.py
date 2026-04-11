"""
Evaluation module for the Yelp Review Summarizer.

Provides two evaluation approaches:
1. Automated metrics — no external API needed (coverage, faithfulness, format compliance)
2. LLM-as-Judge — uses Gemini 2.5 Flash (free tier) to score summaries on multiple dimensions

Usage:
    from evaluate import auto_evaluate, llm_judge_evaluate, run_full_evaluation

    # Automated only
    scores = auto_evaluate(summary, source_reviews)

    # LLM-as-judge only
    scores = llm_judge_evaluate(summary, source_reviews, api_key="...")

    # Both combined
    results = run_full_evaluation(summary, source_reviews, api_key="...")
"""

import re
import json
import time
from typing import Optional


# ===========================
# Approach A: Automated Metrics
# ===========================

# Key themes we expect a good summary to address (based on the prompt's instructions)
EXPECTED_SECTIONS = [
    "overall sentiment",
    "top praised themes",
    "top complaints",
    "representative quotes",
    "confidence level",
]

SENTIMENT_KEYWORDS = {
    "positive": ["great", "excellent", "amazing", "love", "best", "fantastic", "wonderful", "delicious", "friendly", "perfect"],
    "negative": ["bad", "terrible", "worst", "awful", "horrible", "rude", "disgusting", "cold", "slow", "dirty"],
}


def _check_format_compliance(summary: str) -> dict:
    """Check if the summary follows the expected structured format.
    
    Returns:
        dict with 'score' (0-1), 'sections_found', 'sections_missing'
    """
    summary_lower = summary.lower()
    found = []
    missing = []
    
    for section in EXPECTED_SECTIONS:
        # Check for section header (with ## or just the text)
        if section in summary_lower:
            found.append(section)
        else:
            missing.append(section)
    
    score = len(found) / len(EXPECTED_SECTIONS) if EXPECTED_SECTIONS else 0
    return {
        "score": round(score, 2),
        "sections_found": found,
        "sections_missing": missing,
    }


def _check_quote_faithfulness(summary: str, source_reviews: str) -> dict:
    """Check if quoted text in the summary actually appears in the source reviews.
    
    Returns:
        dict with 'score' (0-1), 'total_quotes', 'verified_quotes', 'unverified_quotes'
    """
    # Extract quoted strings from summary (both single and double quotes)
    quotes = re.findall(r'"([^"]{10,})"', summary)
    quotes += re.findall(r"'([^']{10,})'", summary)
    
    if not quotes:
        return {
            "score": 0.5,  # neutral — no quotes to verify
            "total_quotes": 0,
            "verified_quotes": [],
            "unverified_quotes": [],
            "note": "No quotes found in summary to verify"
        }
    
    source_lower = source_reviews.lower()
    verified = []
    unverified = []
    
    for quote in quotes:
        # Check if at least a significant substring of the quote appears in source
        quote_lower = quote.lower().strip()
        # Try matching with progressively shorter substrings
        words = quote_lower.split()
        matched = False
        
        # Try full match first, then 75% of words, then 50%
        for ratio in [1.0, 0.75, 0.5]:
            n_words = max(4, int(len(words) * ratio))
            substring = " ".join(words[:n_words])
            if substring in source_lower:
                matched = True
                break
        
        if matched:
            verified.append(quote)
        else:
            unverified.append(quote)
    
    total = len(quotes)
    score = len(verified) / total if total > 0 else 0
    
    return {
        "score": round(score, 2),
        "total_quotes": total,
        "verified_quotes": verified,
        "unverified_quotes": unverified,
    }


def _check_coverage(summary: str, source_reviews: str) -> dict:
    """Check how well the summary covers themes present in the source reviews.
    
    Returns:
        dict with 'score' (0-1), 'themes_in_source', 'themes_in_summary'
    """
    source_lower = source_reviews.lower()
    summary_lower = summary.lower()
    
    # Common Yelp review themes
    themes = [
        "food", "service", "atmosphere", "price", "wait", "staff",
        "clean", "portion", "flavor", "fresh", "menu", "parking",
        "reservation", "delivery", "ambiance", "decor", "location",
    ]
    
    source_themes = [t for t in themes if t in source_lower]
    summary_themes = [t for t in source_themes if t in summary_lower]
    
    score = len(summary_themes) / len(source_themes) if source_themes else 0
    
    return {
        "score": round(score, 2),
        "themes_in_source": source_themes,
        "themes_in_summary": summary_themes,
        "themes_missed": [t for t in source_themes if t not in summary_themes],
    }


def _check_hallucination_signals(summary: str, source_reviews: str) -> dict:
    """Check for potential hallucination signals in the summary.
    
    Returns:
        dict with 'score' (0-1, higher = less hallucination), 'flags'
    """
    flags = []
    
    # Check if summary mentions "no reviews found" type phrases
    no_review_phrases = ["no reviews found", "no reviews available", "i don't know"]
    for phrase in no_review_phrases:
        if phrase in summary.lower() and source_reviews.strip():
            flags.append(f"Says '{phrase}' but reviews were provided")
    
    # Extract business names from summary and check if they exist in source
    # Look for capitalized multi-word phrases after "Business Name:" in source
    source_businesses = set()
    for match in re.finditer(r"Business Name:\s*([^|]+)", source_reviews):
        name = match.group(1).strip().lower()
        if name and name != "none":
            source_businesses.add(name)
    
    # Check for business names in summary that aren't in source
    # This is a heuristic — look for capitalized proper nouns
    summary_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', summary)
    for name in summary_names:
        name_lower = name.lower()
        # Skip common section headers and generic phrases
        skip_phrases = ["overall sentiment", "top praised", "top complaints", 
                       "representative quotes", "confidence level", "not enough",
                       "yelp reviews", "business name", "review stars"]
        if any(skip in name_lower for skip in skip_phrases):
            continue
        # Check if this name could be a business not in source
        if source_businesses and not any(name_lower in biz for biz in source_businesses):
            # Only flag if it looks like it could be a business name
            if len(name.split()) >= 2:
                flags.append(f"Possible hallucinated business name: '{name}'")
    
    # Score: fewer flags = better
    score = max(0, 1.0 - len(flags) * 0.25)
    
    return {
        "score": round(score, 2),
        "flags": flags,
        "num_flags": len(flags),
    }


def auto_evaluate(summary: str, source_reviews: str) -> dict:
    """Run all automated evaluation metrics.
    
    Args:
        summary: The generated summary text
        source_reviews: The source reviews string (from RAG retrieval)
    
    Returns:
        dict with scores for each metric and an overall score
    """
    format_result = _check_format_compliance(summary)
    quote_result = _check_quote_faithfulness(summary, source_reviews)
    coverage_result = _check_coverage(summary, source_reviews)
    hallucination_result = _check_hallucination_signals(summary, source_reviews)
    
    # Weighted overall score
    overall = (
        format_result["score"] * 0.2 +
        quote_result["score"] * 0.3 +
        coverage_result["score"] * 0.25 +
        hallucination_result["score"] * 0.25
    )
    
    return {
        "overall_auto_score": round(overall, 2),
        "format_compliance": format_result,
        "quote_faithfulness": quote_result,
        "theme_coverage": coverage_result,
        "hallucination_check": hallucination_result,
    }


# ===========================
# Approach B: LLM-as-Judge (Gemini)
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
   - 4: Mostly faithful, minor unsupported details
   - 3: Partially faithful, some unsupported claims
   - 2: Mostly unfaithful, significant fabrication
   - 1: Completely fabricated / hallucinated

2. **Completeness** (Does the summary capture the main themes?)
   - 5: Covers all major positive and negative themes
   - 4: Covers most themes, misses minor ones
   - 3: Covers some themes, misses important ones
   - 2: Very incomplete, misses most themes
   - 1: Does not address the review content at all

3. **Coherence** (Is the summary well-structured and readable?)
   - 5: Perfectly structured, clear, professional
   - 4: Well-structured with minor issues
   - 3: Somewhat organized but confusing in parts
   - 2: Poorly structured, hard to follow
   - 1: Incoherent, unreadable

4. **Relevance** (Does the summary focus on what matters?)
   - 5: Focuses on the most important aspects
   - 4: Mostly relevant, slight tangents
   - 3: Mix of relevant and irrelevant content
   - 2: Mostly irrelevant
   - 1: Completely off-topic

5. **Quote Accuracy** (Are representative quotes actually from the reviews?)
   - 5: All quotes are verbatim from the reviews
   - 4: Quotes are close paraphrases of actual reviews
   - 3: Some quotes match, others appear fabricated
   - 2: Most quotes are fabricated
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


def llm_judge_evaluate(
    summary: str, 
    source_reviews: str, 
    api_key: str,
    model_name: str = "gemini-2.5-flash"
) -> dict:
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
    
    # Truncate source reviews if too long (Gemini has token limits)
    max_source_chars = 30000
    if len(source_reviews) > max_source_chars:
        source_reviews = source_reviews[:max_source_chars] + "\n... [truncated for evaluation]"
    
    user_message = f"""SOURCE REVIEWS:
{source_reviews}

---

GENERATED SUMMARY:
{summary}"""
    
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
        
        # Try to extract JSON from the response
        # Sometimes the model wraps it in ```json ... ```
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response_text)
        
        # Calculate overall judge score (average of all dimensions)
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


# ===========================
# Combined Evaluation
# ===========================

def run_full_evaluation(
    summary: str, 
    source_reviews: str, 
    api_key: Optional[str] = None,
    gemini_model: str = "gemini-2.5-flash",
    print_results: bool = True
) -> dict:
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
    
    # --- Automated metrics ---
    print("Running automated evaluation...") if print_results else None
    results["auto"] = auto_evaluate(summary, source_reviews)
    
    # --- LLM-as-Judge ---
    if api_key:
        print("Running LLM-as-judge evaluation (Gemini)...") if print_results else None
        results["judge"] = llm_judge_evaluate(summary, source_reviews, api_key, gemini_model)
    else:
        results["judge"] = {"skipped": True, "reason": "No API key provided"}
    
    # --- Print formatted report ---
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
