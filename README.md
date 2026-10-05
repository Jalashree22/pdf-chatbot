# 📚 PDF Chatbot – AI Study Assistant

Upload your PDF notes or textbooks and chat with them. The app finds the most relevant parts of your documents and uses Google Gemini to answer, with page-level source citations.

## Features
- **Chat with multiple PDFs** using Retrieval-Augmented Generation (RAG)
- **Source citations** (file name and page number) for every answer
- **Explain simply**: re-explains any answer in beginner-friendly language
- **Smart starter questions** generated from your document
- **Study Mode**: auto-summary, 5-question MCQ quiz with scoring, top-10 key concepts
- **Export** the whole chat as a text file

## Tech Stack
Python · Streamlit · LangChain · Google Gemini (`gemini-2.5-flash`) · FAISS · HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`) · PyPDF

## How it works
1. PDFs are read page by page and split into overlapping 1000-character chunks.
2. Each chunk is converted into an embedding (`all-MiniLM-L6-v2`) and stored in a FAISS vector index.
3. For each question, the 4 most similar chunks are retrieved.
4. Those chunks plus recent chat history are sent to Gemini, which answers only from that context.

## Project Structure
```
pdf-chatbot/
├── app.py                  # Streamlit user interface
├── requirements.txt
├── .env.example            # shows where to put your API key
└── utils/
    ├── pdf_processor.py    # PDF loading, chunking, FAISS index
    ├── llm_handler.py      # Gemini calls: answers, simple explanations, suggestions
    └── student_tools.py    # summary, quiz, key concepts, chat export
```

## Setup
1. Clone the repo and open the folder:
   ```
   git clone https://github.com/Jalashree22/pdf-chatbot.git
   cd pdf-chatbot
   ```
2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
4. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
5. Copy `.env.example` to `.env` and paste your key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
6. Run the app:
   ```
   streamlit run app.py
   ```
   The first run downloads the embedding model (about 90 MB), so it may take a minute.


