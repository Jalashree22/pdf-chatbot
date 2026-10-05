"""
llm_handler.py
Manages all Gemini LLM interactions, prompt templates, and response parsing.
"""

import os
from typing import Tuple, List

from langchain_google_genai import ChatGoogleGenerativeAI


"""MODEL_NAME = "models/gemini-1.5-pro-latest"
TEMPERATURE = 0.3"""

MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.3


def get_llm(temperature: float = TEMPERATURE) -> ChatGoogleGenerativeAI:
    """
    Instantiate and return the Gemini LLM client.

    Args:
        temperature: Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Please add it to your .env file."
        )
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=api_key,
    )


def answer_question(
    question: str,
    context_docs: List,
    chat_history: List[dict],
) -> Tuple[str, List[dict]]:
    """
    Answer a user question using retrieved context and conversation history.

    Args:
        question: The user's question.
        context_docs: List of relevant LangChain Document chunks from FAISS.
        chat_history: List of previous {"role": ..., "content": ...} messages.

    Returns:
        Tuple of (answer string, list of source citation dicts).
    """
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source_file','?')}, Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
        for doc in context_docs
    )

    history_text = ""
    for msg in chat_history[-6:]:  # Last 3 exchanges for context window efficiency
        role = "Student" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful academic assistant for students. Answer the question clearly and accurately based ONLY on the provided context. If the answer is not in the context, say so honestly.

Previous conversation:
{history_text}

Context from uploaded documents:
{context}

Student's question: {question}

Provide a clear, well-structured answer. Use bullet points or numbered lists when listing multiple points."""

    llm = get_llm()
    response = llm.invoke(prompt)

    # Build source citations
    sources = []
    seen = set()
    for doc in context_docs:
        source_file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            sources.append({"file": source_file, "page": page})

    return response.content, sources


def explain_simply(answer: str) -> str:
    """
    Re-explain a given answer in very simple, beginner-friendly language.

    Args:
        answer: The original detailed answer text.

    Returns:
        Simplified explanation string.
    """
    prompt = f"""Explain the following answer as if you're talking to a 12-year-old student who is new to this topic. Use simple words, relatable analogies, and short sentences. Avoid jargon.

Original answer:
{answer}

Simple explanation:"""

    llm = get_llm(temperature=0.5)
    response = llm.invoke(prompt)
    return response.content


def generate_smart_suggestions(sample_text: str) -> List[str]:
    """
    Generate 4 smart starter questions based on document content.

    Args:
        sample_text: A sample of the document text (first ~2000 chars).

    Returns:
        List of 4 question strings.
    """
    prompt = f"""Based on the following text from a student's document, generate exactly 4 short, useful questions a student might want to ask. Make them practical and specific to the content. Return ONLY the 4 questions, one per line, no numbering or bullet points.

Document excerpt:
{sample_text[:2000]}

4 questions:"""

    llm = get_llm(temperature=0.6)
    response = llm.invoke(prompt)
    lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return lines[:4]
