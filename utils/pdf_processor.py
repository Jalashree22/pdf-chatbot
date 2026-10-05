"""
pdf_processor.py
Handles PDF loading, text chunking, and FAISS vectorstore creation.
"""

import tempfile
import os
from typing import List, Tuple, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
WORDS_PER_MINUTE = 200  # Average reading speed


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load and return HuggingFace sentence-transformer embeddings.
    Cached via Streamlit session to avoid reloading.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_pdf(uploaded_file) -> Tuple[List, Dict]:
    """
    Load a single uploaded PDF file and extract documents + metadata.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Tuple of (list of LangChain Document objects, metadata dict).

    Raises:
        ValueError: If the PDF is empty or unreadable.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
    finally:
        os.unlink(tmp_path)

    if not documents:
        raise ValueError(f"'{uploaded_file.name}' appears to be empty or unreadable.")

    total_pages = len(documents)
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    reading_time = max(1, round(total_words / WORDS_PER_MINUTE))

    # Tag each document chunk with its source filename
    for doc in documents:
        doc.metadata["source_file"] = uploaded_file.name

    metadata = {
        "filename": uploaded_file.name,
        "total_pages": total_pages,
        "total_words": total_words,
        "reading_time_minutes": reading_time,
    }

    return documents, metadata


def process_pdfs(uploaded_files) -> Tuple[FAISS, List[Dict], str]:
    """
    Process multiple uploaded PDF files into a single merged FAISS vectorstore.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of (FAISS vectorstore, list of metadata dicts, combined raw text).

    Raises:
        ValueError: If no valid documents are extracted.
    """
    all_documents = []
    all_metadata = []
    all_text_parts = []

    for uploaded_file in uploaded_files:
        if not uploaded_file.name.lower().endswith(".pdf"):
            raise ValueError(f"'{uploaded_file.name}' is not a PDF file.")

        documents, metadata = load_pdf(uploaded_file)
        all_documents.extend(documents)
        all_metadata.append(metadata)
        all_text_parts.append(
            "\n".join(doc.page_content for doc in documents)
        )

    if not all_documents:
        raise ValueError("No content could be extracted from the uploaded files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    combined_text = "\n\n---\n\n".join(all_text_parts)

    return vectorstore, all_metadata, combined_text


def similarity_search(vectorstore: FAISS, query: str, k: int = 4) -> List:
    """
    Retrieve top-k relevant document chunks for a query.

    Args:
        vectorstore: FAISS vectorstore.
        query: User question string.
        k: Number of chunks to retrieve.

    Returns:
        List of LangChain Document objects.
    """
    return vectorstore.similarity_search(query, k=k)
