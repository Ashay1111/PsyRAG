RAG for Cognitive Psychology using Gemini
=========================================

This project is an advanced Retrieval-Augmented Generation (RAG) system built with LangChain, Gemini 2.0, and FAISS. Going beyond basic RAG implementations, it features intelligent query expansion that automatically generates semantic variations of user queries and parallel retrieval processing for enhanced accuracy and performance. The system applies sophisticated semantic search and natural language generation to content from psychology books and papers (You can change the genre!)— including Daniel Kahneman's "Thinking, Fast and Slow", one of my favourite book!

Use Case
--------

The goal is to build an intelligent assistant that can:
- Ingest and index domain-specific literature (PDFs)
- Automatically expand user queries into multiple semantic variations for comprehensive retrieval
- Retrieve relevant information using parallel processing across multiple query formulations
- Generate human-like answers grounded in source material

Project Structure
-----------------
### Project Structure

```text
PsyRAG/
├── books/                (psychology books in PDF)
├── research_papers/      (research papers in PDF)
├── data/                 (stores FAISS index)
├── scripts/
│   ├── .env              (environment variables - create this)
│   ├── main.py           (main application entry point)
│   ├── config.py         (paths, constants)
│   ├── query_expansion.py(query reformulation for better retrieval)
│   ├── indexing.py       (chunk + embed)
│   ├── retrieval.py      (semantic search)
│   └── generation.py     (LLM response)
├── requirements.txt
├── .gitignore
└── README.md

```

How to Run
----------

## 1. Install requirements
```bash
pip install -r requirements.txt
```

## 2. Set up environment variables
Create a `.env` file at the root:
```
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

## 3. Prepare your PDFs
- Add books to `books/` folder
- Add papers to `research_papers/` folder

## 4. Index the documents
```bash
cd scripts
python indexing.py
```

## 5. Test retrieval
```bash
python retrieval.py
```

## 6. Generate answers
```bash
python generation.py
```

## 7. Run the main application
```bash
python main.py
```


## Notes:
- Make sure your PDF folders exist before running the indexing script
- The indexing step will create a `data/faiss_index` folder to store the vector database
- All Python files should be run from the root directory of the project

Tech Stack
----------

LangChain                → document loading, chunking, embedding  
Google Generative AI SDK → Gemini 2.0 for embedding + generation  
FAISS                    → local vector database for retrieval  
PyMuPDF / PDFMiner       → PDF text extraction

Configuration Notes
-------------------

- Embeddings use Gemini's model: "models/embedding-001"
- Text generation uses: "gemini-2.0-flash" with temperature=0 for consistent outputs
- FAISS saves index locally at data/faiss_index/
- Document chunking: 800 characters with 150 character overlap for context preservation
- Query expansion generates 3 semantic variations per query for comprehensive retrieval
- Parallel retrieval with configurable thread pool (default: 4 workers)

Limitations
-----------

- Local only — no hosted retrieval or generation
- Works best with clean, OCRed PDFs
- Chunking is semantic but not structure-aware (e.g., chapters, sections)
- Query expansion relies on LLM

Why This Project Matters
------------------------

This project demonstrates how to build a clean, real-world RAG system over a focused domain — cognitive psychology — using modern tooling. It showcases:

- Advanced Query Processing: Intelligent query expansion with semantic variation generation for better retrieval
- Parallel Computing: Concurrent document retrieval across multiple query formulations with configurable thread pools
- Deduplication: Advanced content hashing to eliminate redundant retrieved passages
- Modular Architecture: Clean separation of concerns with dedicated modules for indexing, retrieval, and generation
- Robust PDF Processing: Multiple fallback mechanisms for reliable document ingestion
- Real-world Application: Practical implementation of Gemini in an LLM-powered pipeline
- Scalable Design: Efficient vector storage and parallel processing for performance optimization