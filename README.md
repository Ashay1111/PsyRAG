RAG for Cognitive Psychology using Gemini
=========================================

This project is a modular Retrieval-Augmented Generation (RAG) system built with LangChain, Gemini 2.0, and FAISS. It applies semantic search and natural language generation to content from psychology books and papers (You can change that)— including Daniel Kahneman's "Thinking, Fast and Slow", one of my favourite book!

Use Case
--------

The goal is to build an intelligent assistant that can:
- Ingest and index domain-specific literature (PDFs)
- Retrieve relevant information in response to a query
- Generate human-like answers grounded in source material

Project Structure
-----------------

rag-psychology-gemini/
├── books/                ← psychology books (PDFs)
├── research_papers/      ← research papers (PDFs)
├── data/                 ← stores FAISS index
├── scripts/
│   ├── config.py       
│   ├── indexing.py       ← loads, chunks, embeds, and indexes
│   ├── retrieval.py      ← retrieves top-k relevant chunks
│   └── generation.py     ← prompts Gemini to answer with context
├── requirements.txt
├── .gitignore
└── README.md

How to Run
----------

1. Install requirements

    pip install -r requirements.txt

2. Set up environment variables

    Create a `.env` file at the root:

    GOOGLE_API_KEY=your_google_generative_ai_key

    (Alternatively, use system-level Google Cloud authentication)

3. Prepare your PDFs

    - Add books to `books/`
    - Add papers to `papers/`

4. Index the documents

    cd scripts
    python indexing.py

5. Test retrieval

    python retrieval.py

6. Generate answers

    python generation.py

Tech Stack
----------

LangChain                → document loading, chunking, embedding  
Google Generative AI SDK → Gemini 2.0 for embedding + generation  
FAISS                    → local vector database for retrieval  
PyMuPDF / PDFMiner       → PDF text extraction

Configuration Notes
-------------------

- Embeddings use Gemini's model: "models/embedding-001"
- FAISS saves index locally at `data/faiss_index/`
- Prompt template uses simple instruction-following format

Limitations
-----------

- Local only — no hosted retrieval or generation
- Works best with clean, OCRed PDFs
- Chunking is semantic but not structure-aware (e.g., chapters, sections)

Why This Project Matters
------------------------

This project demonstrates how to build a clean, real-world RAG system over a focused domain — cognitive psychology — using modern tooling. It showcases:

- Modular code architecture
- Reusable components
- Practical application of Gemini in an LLM-powered pipeline
- Strong semantic search with explainable sources

Perfect for showcasing both backend engineering skills and thoughtful domain-specific use of AI.
