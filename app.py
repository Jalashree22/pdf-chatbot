"""
app.py
Streamlit front-end for the PDF Chatbot (Study Assistant).

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from the .env file

from utils.pdf_processor import process_pdfs, similarity_search  # noqa: E402
from utils.llm_handler import (  # noqa: E402
    answer_question,
    explain_simply,
    generate_smart_suggestions,
)
from utils.student_tools import (  # noqa: E402
    summarize_documents,
    generate_quiz,
    extract_key_concepts,
    export_chat_as_text,
)

st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide")


# ---------------------------------------------------------------- state
def init_state():
    defaults = {
        "vectorstore": None,
        "metadata": [],
        "combined_text": "",
        "chat_history": [],
        "suggestions": [],
        "pending_question": None,
        "summary": None,
        "quiz": None,
        "concepts": None,
        "processed_names": (),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_study_outputs():
    st.session_state.chat_history = []
    st.session_state.suggestions = []
    st.session_state.pending_question = None
    st.session_state.summary = None
    st.session_state.quiz = None
    st.session_state.concepts = None


init_state()


# -------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 PDF Chatbot")
    st.caption("Upload your notes and chat with them.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
        with st.spinner("Reading and indexing your PDFs..."):
            try:
                vs, meta, text = process_pdfs(uploaded_files)
                reset_study_outputs()
                st.session_state.vectorstore = vs
                st.session_state.metadata = meta
                st.session_state.combined_text = text
                st.session_state.processed_names = tuple(f.name for f in uploaded_files)
                try:
                    st.session_state.suggestions = generate_smart_suggestions(text)
                except Exception:
                    st.session_state.suggestions = []  # suggestions are optional
                st.success("PDFs ready! Ask your questions.")
            except Exception as e:
                st.error(f"Could not process the PDFs: {e}")

    if st.session_state.metadata:
        st.subheader("Loaded documents")
        for m in st.session_state.metadata:
            st.markdown(
                f"**{m['filename']}**  \n"
                f"{m['total_pages']} pages · {m['total_words']} words · "
                f"~{m['reading_time_minutes']} min read"
            )

        st.download_button(
            "⬇️ Export chat",
            data=export_chat_as_text(
                st.session_state.chat_history, st.session_state.metadata
            ),
            file_name="study_session.txt",
            mime="text/plain",
            disabled=not st.session_state.chat_history,
        )
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ----------------------------------------------------------------- main
st.title("📚 Chat with your PDFs")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process PDFs** to begin.")
    st.stop()

tab_chat, tab_study = st.tabs(["💬 Chat", "🎓 Study Mode"])

# ------------------------------------------------------------ chat tab
with tab_chat:
    if st.session_state.suggestions and not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggestions):
            if cols[i % 2].button(q, key=f"sugg_{i}"):
                st.session_state.pending_question = q
                st.rerun()

    # Show conversation so far
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    st.caption(
                        "📎 Sources: "
                        + ", ".join(f"{s['file']} (p.{s['page']})" for s in msg["sources"])
                    )
                if st.button("🧒 Explain simply", key=f"simple_{idx}"):
                    with st.spinner("Simplifying..."):
                        try:
                            st.markdown(explain_simply(msg["content"]))
                        except Exception as e:
                            st.error(f"Error: {e}")

    typed = st.chat_input("Ask a question about your PDFs...")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if question:
        history_before = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    docs = similarity_search(st.session_state.vectorstore, question)
                    answer, sources = answer_question(question, docs, history_before)
                except Exception as e:
                    answer, sources = f"⚠️ Sorry, something went wrong: {e}", []
            st.markdown(answer)
            if sources:
                st.caption(
                    "📎 Sources: "
                    + ", ".join(f"{s['file']} (p.{s['page']})" for s in sources)
                )

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()

# ----------------------------------------------------------- study tab
with tab_study:
    c1, c2, c3 = st.columns(3)

    if c1.button("📝 Summarize", use_container_width=True):
        with st.spinner("Writing summary..."):
            try:
                st.session_state.summary = summarize_documents(
                    st.session_state.combined_text
                )
            except Exception as e:
                st.error(f"Error: {e}")

    if c2.button("❓ Generate Quiz", use_container_width=True):
        with st.spinner("Creating quiz..."):
            try:
                st.session_state.quiz = generate_quiz(st.session_state.combined_text)
                if not st.session_state.quiz:
                    st.warning("Could not build a quiz this time. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")

    if c3.button("🔑 Key Concepts", use_container_width=True):
        with st.spinner("Extracting concepts..."):
            try:
                st.session_state.concepts = extract_key_concepts(
                    st.session_state.combined_text
                )
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.summary:
        st.subheader("Summary")
        st.markdown(st.session_state.summary)

    if st.session_state.concepts:
        st.subheader("Key Concepts")
        for c in st.session_state.concepts:
            st.markdown(f"**{c['term']}**: {c['definition']}")

    if st.session_state.quiz:
        st.subheader("Quiz")
        with st.form("quiz_form"):
            picks = []
            for i, q in enumerate(st.session_state.quiz):
                picks.append(
                    st.radio(
                        f"Q{i + 1}. {q['question']}",
                        options=list(q["options"].keys()),
                        format_func=lambda k, q=q: f"{k}) {q['options'][k]}",
                        index=None,
                        key=f"quiz_q_{i}",
                    )
                )
            submitted = st.form_submit_button("Check answers")

        if submitted:
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                correct = q["answer"][:1]
                if picks[i] == correct:
                    score += 1
                    st.success(f"Q{i + 1}: Correct! {q['explanation']}")
                else:
                    st.error(
                        f"Q{i + 1}: Wrong. Correct answer: {correct}) "
                        f"{q['options'].get(correct, '')}. {q['explanation']}"
                    )
            st.markdown(f"### Score: {score} / {len(st.session_state.quiz)}")
