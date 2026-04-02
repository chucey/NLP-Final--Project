"""
This file builds a RAG (Retrieval-Augmented Generation) system using FAISS and a language model. The RAG system is designed to retrieve relevant information from a knowledge base and generate responses based on that information. The code in this file is intended to be run as part of the main data loading and preprocessing pipeline, and it assumes that the necessary data has already been prepared and saved in the appropriate format. The RAG system can be used for various applications, such as question answering, summarization, or any task that benefits from combining retrieval and generation capabilities.
"""
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_index(
    csv_path: str,
    index_dir: str = "faiss_yelp",
    text_col: str = "clean_text",
    metadata_cols = ("review_id", "business_id", "business_name", "categories", "review_stars", "date") 
):
    """Builds a FAISS index from a CSV file containing text data and metadata.

    Args:
        csv_path (str): Path to the CSV file containing the data.
        index_dir (str, optional): Directory to save the FAISS index. Defaults to "RAG/faiss_yelp".
        text_col (str, optional): Column name for the text data. Defaults to "clean_text".
        metadata_cols (tuple, optional): Column names for the metadata. Defaults to ("review_id", "business_id", "business_name", "categories", "review_stars", "date").
    """
    df = pd.read_csv(csv_path).dropna(subset=[text_col])

    docs = []
    for _, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text:
            continue

        meta = {}
        for col in metadata_cols:
            if col in df.columns:
                value = row[col]
                if pd.isna(value):
                    value = None
                if col == "review_stars" and value is not None:
                    value = int(value)
                meta[col] = value

        docs.append(Document(page_content=text, metadata=meta))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunked_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunked_docs, embeddings)
    vs.save_local(index_dir)
    print(f"Saved {len(chunked_docs)} chunks to {index_dir}")

if __name__ == "__main__":
    build_index("data/sample_reviews_dataset_200.csv")