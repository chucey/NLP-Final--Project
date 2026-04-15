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

        if actual is None:
            return False

        # ===== dict =====
        if isinstance(expected, dict):
            try:
                actual_num = float(actual)
                val_num = float(expected.get("value"))
            except (TypeError, ValueError):
                return False

            op = expected.get("op")

            if op == "lt":
                if not (actual_num < val_num):
                    return False
            elif op == "lte":
                if not (actual_num <= val_num):
                    return False
            elif op == "gt":
                if not (actual_num > val_num):
                    return False
            elif op == "gte":
                if not (actual_num >= val_num):
                    return False
            elif op == "eq":
                if not (actual_num == val_num):
                    return False
            else:
                return False

            # This filter key has already been evaluated.
            continue

        # Use case-insensitive substring matching for text metadata.
        if isinstance(expected, str):
            if _normalize(expected) not in _normalize(actual):
                return False
        else:
            if actual != expected:
                return False
    return True

def load_vectorstore(index_dir: str ="faiss_yelp",
                     model: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    loads a FAISS vectorstore

    Args:
        index_dir (str, optional): directory of the vectorstore. Defaults to "faiss_yelp".
        model (str, optional): name of the embedding model to use. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
        FAISS: the loaded FAISS vectorstore
    """

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"token": hf_token} if hf_token else {}
    emb = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs=model_kwargs,
    )
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

def retrieve_reviews_for_summary(vs: FAISS,
                                 metadata_filter: dict = None,
                                 query: str = None,
                                 k: int = 80,
                                 eval_mode: bool = False) -> str | set[str]:
    """
    Retrieves documents from the FAISS vectorstore based on metadata filtering or a similarity search. If any metadata filters are specified, it retrieves all documents from the FAISS docstore and applies the filters in Python to find matching documents. If no metadata filters are provided, it performs a similarity search using the provided query. If no query or metadata filters are provided, the function outputs "no results found." The retrieved documents are then formatted into a string that includes selected metadata fields and the content of the reviews.

    If eval_mode is True, the function will return a set of retrieved review IDs instead of the formatted string. This is used for evaluating retrieval performance against ground truth labels.

    Args:
        vs (FAISS): the loaded FAISS vectorstore to search for relevant documents
        metadata_filter (dict, optional): A dictionary of metadata fields and their expected values to filter the documents. For string fields, the document's metadata value must contain the expected value as a substring (case-insensitive). For non-string fields, an exact match is required. Defaults to None.
            The metadata fields can include:
            - business_name (str): The name of the business to filter by.
            - city (str): The city to filter by.
            - state (str): The state to filter by.
            - review_stars (int): The number of stars to filter by.
            - categories (str): The categories to filter by.

            The expected values can be specified as follows:
            - For string fields, provide a substring to match (e.g. "Italian" to match any category containing "Italian").
            - For numeric fields, you can provide an exact value (e.g. 5 to match reviews with 5 stars).
            - For numeric fields, you can also provide a dictionary with an operator and value for range filtering (e.g. {"op": "gte", "value": 4} to match reviews with 4 or more stars).
                The supported operators are:
                - "lt": less than
                - "lte": less than or equal to
                - "gt": greater than
                - "gte": greater than or equal to
                - "eq": equal to

            Example metadata_filter:
            {
                "business_name": "Home Depot",
                "city": "Phoenix",
                "state": "AZ",
                "review_stars": {"op": "gte", "value": 4},
                "categories": "Hardware"
            } 
                            or
             {
                "business_name": "Home Depot",
                "city": "Phoenix",
                "state": "AZ",
                "review_stars": 4,
                "categories": "Hardware"
            } 

        query (str, optional): The query to use for similarity search. Defaults to None.
        k (int, optional): The number of documents to return. Defaults to 80.
        eval_mode (bool, optional): Whether to return a set of retrieved review IDs for evaluation purposes instead of the formatted string. Defaults to False.

    Returns:
        str | set[str]: a formatted string containing the retrieved reviews and selected metadata, or a set of retrieved review IDs if eval_mode is True
    """
   

    # If metadata is provided, fetch matching docs directly from the FAISS docstore.
    # This is more reliable for summary workloads than semantic pre-filtering.

    metadata_filter = metadata_filter or {}

    if all(value is None for value in metadata_filter.values()) and not query:
        return "No results found."
    if all(value is None for value in metadata_filter.values()) and query:
        matches = vs.similarity_search(query=query, k=k)
    else: 
        matches = []
        metadata_filter = {k: v for k, v in metadata_filter.items() if v is not None}  # remove None values from filter
        if metadata_filter:
            all_docs = getattr(vs.docstore, "_dict", {}).values()
            matches = [doc for doc in all_docs if _doc_matches_filters(doc, metadata_filter)]
            if not matches and not query:
                return "no results found."
            elif not matches:
                # This avoids empty results from tiny typos/casing differences in exact filter matching.
                candidate_k = max(k * 8, 100)
                candidates = vs.similarity_search(query=query, k=candidate_k)
                matches = [doc for doc in candidates if _doc_matches_filters(doc, metadata_filter)]

        else:      
            matches = vs.similarity_search(query=query, k=k)

    if eval_mode:
        return {doc.metadata.get('review_id') for doc in matches[:k]}

    formatted_reviews = [
        f"Business Name: {doc.metadata.get('business_name')} | Content: {doc.page_content} | City: {doc.metadata.get('city')} | State: {doc.metadata.get('state')} | Review Stars: {doc.metadata.get('review_stars')}\n---"
        for doc in matches[:k]
    ]
    if not formatted_reviews:
        return "No results found."

    return "\n\n".join(formatted_reviews)

# if __name__ == "__main__":
#     vs = load_vectorstore()
#     metadata_filter = {
#         "categories": None,
#         'business_name': 'Home Depot',
#         "city": None,
#         "state": None,
#         "review_stars": {"op": "gte", "value": 4}
#     }
#     results = retrieve_reviews_for_summary(vs, 
#                                            metadata_filter = metadata_filter,
#                                             query='good breakfast spots',
#                                             k=10,
#                                             eval_mode=True)
#     print(results)

