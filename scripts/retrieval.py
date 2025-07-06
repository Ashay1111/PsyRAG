import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import gemini_api_key, google_api_key

# Load FAISS index from disk
def load_faiss_index(index_path="../data/faiss_index"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Perform semantic search
def retrieve_chunks(query: str, k: int = 5):
    index = load_faiss_index()
    docs = index.similarity_search(query, k=k)
    return docs

# Test the retrieval
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve_chunks(query, k=5)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
        print(f"[source: {doc.metadata.get('filename')}]")
