"""
student_tools.py
Implements Study Mode features: Auto-Summarize, Quiz Generator, Key Concepts Extractor.
"""

import re
from typing import List, Dict

from utils.llm_handler import get_llm


MAX_TEXT_LENGTH = 12000  # Truncate to stay within token limits


def _truncate(text: str, max_len: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to a safe length for LLM context windows."""
    return text[:max_len] + ("\n\n[...document truncated for processing...]" if len(text) > max_len else "")


def summarize_documents(combined_text: str) -> str:
    """
    Generate a structured academic summary of all uploaded documents.

    Args:
        combined_text: Full concatenated text from all PDFs.

    Returns:
        A multi-section summary string.
    """
    truncated = _truncate(combined_text)

    prompt = f"""You are an academic assistant. Create a comprehensive study summary of the following document(s) for a student. Structure your response with these sections:

📌 **Overview** (2-3 sentences about what the document is about)

🎯 **Main Topics Covered** (bullet list of key topics)

📝 **Key Points** (5-8 most important points a student should remember)

💡 **Important Definitions** (any key terms defined in the text)

🔗 **Connections & Themes** (how the topics relate to each other)

Document text:
{truncated}

Provide a thorough but student-friendly summary:"""

    llm = get_llm(temperature=0.3)
    response = llm.invoke(prompt)
    return response.content


def generate_quiz(combined_text: str) -> List[Dict]:
    """
    Generate 5 multiple-choice questions from the document content.

    Args:
        combined_text: Full concatenated text from all PDFs.

    Returns:
        List of dicts with keys: question, options (list of 4), answer, explanation.
    """
    truncated = _truncate(combined_text)

    prompt = f"""Create exactly 5 multiple-choice quiz questions based on the following document. These should test genuine understanding, not just memorization.

Format your response as follows (follow this format strictly):
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
ANSWER: [Correct letter]
EXPLANATION: [Brief explanation why this is correct]

Q2: ...and so on for all 5 questions.

Document text:
{truncated}

5 MCQ Questions:"""

    llm = get_llm(temperature=0.4)
    response = llm.invoke(prompt)

    return _parse_quiz(response.content)


def _parse_quiz(raw: str) -> List[Dict]:
    """
    Parse raw LLM quiz output into structured question dicts.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        List of parsed question dicts.
    """
    questions = []
    blocks = re.split(r'\n(?=Q\d+:)', raw.strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if not lines:
            continue
        try:
            q_text = re.sub(r'^Q\d+:\s*', '', lines[0])
            options = {}
            answer = ""
            explanation = ""

            for line in lines[1:]:
                if re.match(r'^[A-D]\)', line):
                    letter = line[0]
                    options[letter] = line[3:].strip()
                elif line.upper().startswith("ANSWER:"):
                    answer = line.split(":", 1)[1].strip().upper()
                elif line.upper().startswith("EXPLANATION:"):
                    explanation = line.split(":", 1)[1].strip()

            if q_text and len(options) == 4 and answer:
                questions.append({
                    "question": q_text,
                    "options": options,
                    "answer": answer,
                    "explanation": explanation,
                })
        except Exception:
            continue  # Skip malformed blocks

    return questions


def extract_key_concepts(combined_text: str) -> List[Dict]:
    """
    Extract the top 10 key concepts/terms from the document.

    Args:
        combined_text: Full concatenated text from all PDFs.

    Returns:
        List of dicts with 'term' and 'definition' keys.
    """
    truncated = _truncate(combined_text)

    prompt = f"""Extract the 10 most important concepts, terms, or topics from the following document that a student absolutely must understand.

Format your response exactly like this:
TERM: [Term or concept name]
DEFINITION: [Clear, concise explanation in 1-2 sentences]

Repeat for all 10 terms.

Document text:
{truncated}

Top 10 Key Concepts:"""

    llm = get_llm(temperature=0.2)
    response = llm.invoke(prompt)

    return _parse_concepts(response.content)


def _parse_concepts(raw: str) -> List[Dict]:
    """
    Parse raw LLM concept output into structured dicts.

    Args:
        raw: Raw string output from the LLM.

    Returns:
        List of dicts with 'term' and 'definition'.
    """
    concepts = []
    blocks = re.split(r'\n(?=TERM:)', raw.strip())

    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        term = ""
        definition = ""
        for line in lines:
            if line.upper().startswith("TERM:"):
                term = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DEFINITION:"):
                definition = line.split(":", 1)[1].strip()
        if term and definition:
            concepts.append({"term": term, "definition": definition})

    return concepts[:10]


def export_chat_as_text(chat_history: List[Dict], metadata_list: List[Dict]) -> str:
    """
    Export the full chat conversation as a formatted plain-text string.

    Args:
        chat_history: List of {"role": ..., "content": ..., "sources": ...} dicts.
        metadata_list: List of document metadata dicts.

    Returns:
        Formatted string ready to save as .txt.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("📚 PDF CHATBOT - STUDY SESSION EXPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Documents studied:")
    for m in metadata_list:
        lines.append(f"  • {m['filename']} ({m['total_pages']} pages, ~{m['reading_time_minutes']} min read)")
    lines.append("")
    lines.append("-" * 60)
    lines.append("CONVERSATION")
    lines.append("-" * 60)
    lines.append("")

    for msg in chat_history:
        if msg["role"] == "user":
            lines.append(f"🧑 YOU: {msg['content']}")
        else:
            lines.append(f"🤖 ASSISTANT: {msg['content']}")
            if msg.get("sources"):
                src_str = ", ".join(
                    f"{s['file']} p.{s['page']}" for s in msg["sources"]
                )
                lines.append(f"   📎 Sources: {src_str}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("End of session")
    lines.append("=" * 60)

    return "\n".join(lines)
