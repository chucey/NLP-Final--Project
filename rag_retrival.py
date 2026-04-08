"""
This file contains code for retriving data from the FAISS index built in build_rag.py. The retrieval process involves loading the FAISS index, performing a similarity search based on a query, and returning the relevant documents along with their metadata. This code is intended to be used as part of a larger RAG system, where the retrieved information can be fed into a language model for generating responses or performing other tasks that require access to the knowledge base. The retrieval functionality is crucial for enabling the RAG system to provide accurate and contextually relevant information based on user queries.
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os


def _normalize(value: str) -> str:
    """
    Create a normalized version of the string for more robust matching (e.g. case-insensitive, ignore leading/trailing whitespace)

    Args:
        value (str): the string to normalize

    Returns:
        str: the normalized string, or empty string if the input is None
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def _doc_matches_filters(doc: Document, metadata_filter: dict) -> bool:
    """
    Check if a document's metadata matches the provided filter criteria. For string metadata, use case-insensitive substring matching to allow for more flexible filtering.

    Args:
        doc (Document): the document whose metadata is being checked against the filter criteria
        metadata_filter (dict): the filter criteria, where keys are metadata fields and values are the expected values to match. For string fields, the document's metadata value must contain the expected value as a substring (case-insensitive). For non-string fields, an exact match is required.

    Returns:
        bool: True if the document's metadata matches all filter criteria, False otherwise
    """
    for key, expected in metadata_filter.items():
        actual = doc.metadata.get(key)

        # Use case-insensitive substring matching for text metadata.
        if isinstance(expected, str):
            if _normalize(expected) not in _normalize(actual):
                return False
        else:
            if actual != expected:
                return False
    return True

def load_vectorstore(index_dir: str ="faiss_yelp") -> FAISS:
    """
    loads a FAISS vectorstore

    Args:
        index_dir (str, optional): directory of the vectorstore. Defaults to "faiss_yelp".

    Returns:
        FAISS: the loaded FAISS vectorstore
    """

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"token": hf_token} if hf_token else {}
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
    )
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

def retrieve_reviews_for_summary(vs: FAISS,
                                 business_name: str = None,
                                 city: str = None,
                                 state: str = None,
                                 review_stars: int = None,
                                 categories: str = None,
                                 k: int = 80) -> str:
    """
    Retrieves documents from the FAISS vectorstore based on a broad query related to customer experience, food quality, service, atmosphere, value, wait time, and cleanliness. The retrieval can be optionally filtered by business name, review stars, and categories. If metadata filters are provided, the function first attempts to fetch matching documents directly from the FAISS docstore for more reliable results. If no metadata filters are provided or if the direct fetch yields no results, it performs a similarity search using the broad query and then applies metadata filtering in Python to ensure robust matching.

    Args:
        vs (FAISS): the loaded FAISS vectorstore to search for relevant documents
        business_name (str, optional): The name of the business to filter by. Defaults to None.
        city (str, optional): The city to filter by. Defaults to None.
        state (str, optional): The state to filter by. Defaults to None.
        review_stars (int, optional): The number of stars to filter by. Defaults to None.
        categories (str, optional): The categories to filter by. Defaults to None.
        k (int, optional): The number of documents to return. Defaults to 80.

    Returns:
        str: a formatted string containing the retrieved reviews and selected metadata
    """
    metadata_filter = {}
    
    if business_name:
        metadata_filter["business_name"] = business_name
    if review_stars:
        metadata_filter["review_stars"] = review_stars
    if categories:
        metadata_filter["categories"] = categories
    if city:
        metadata_filter["city"] = city
    if state:
        metadata_filter["state"] = state

    # If metadata is provided, fetch matching docs directly from the FAISS docstore.
    # This is more reliable for summary workloads than semantic pre-filtering.
    matches = []
    if metadata_filter:
        all_docs = getattr(vs.docstore, "_dict", {}).values()
        matches = [doc for doc in all_docs if _doc_matches_filters(doc, metadata_filter)]
        if not matches:
            # broad "summary-oriented" query
            query = "overall customer experience food quality service atmosphere value wait time cleanliness"

            # Pull a broader candidate set then apply robust metadata checks in Python.
            # This avoids empty results from tiny typos/casing differences in exact filter matching.
            candidate_k = max(k * 8, 100)
            candidates = vs.similarity_search(query=query, k=candidate_k)
            matches = [doc for doc in candidates if _doc_matches_filters(doc, metadata_filter)]

    else:
        # broad "summary-oriented" query
        query = "overall customer experience food quality service atmosphere value wait time cleanliness"
        matches = vs.similarity_search(query=query, k=k)

    formatted_reviews = [
        f"Business Name: {doc.metadata.get('business_name')} | Content: {doc.page_content} | City: {doc.metadata.get('city')} | State: {doc.metadata.get('state')} | Review Stars: {doc.metadata.get('review_stars')}\n---"
        for doc in matches[:k]
    ]
    return "\n\n".join(formatted_reviews)

# vs = load_vectorstore()
# results = retrieve_reviews_for_summary(vs, categories="Italian", k=10)
# print(results)

# if __name__ == "__main__":
#     vs = load_vectorstore()
#     results = retrieve_reviews_for_summary(vs, categories="Italian", k=10)
#     # print(f"Retrieved {len(results)} review chunks")
#     print("Sample retrieved document metadata and content:")
#     print(results)
#     # print(f"Retrieved {len(results)} review chunks")
#     # for doc in results:
#     #     print(doc.metadata)
#     #     print(doc.page_content[:200])  # print first 200 chars of the review
#     #     print("-" * 80)