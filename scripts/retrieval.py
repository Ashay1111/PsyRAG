import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import gemini_api_key, google_api_key
import concurrent.futures

def retrieve_multiple_queries(queries, retriever, max_workers=4):
    """Retrieve relevant documents for multiple queries in parallel."""
    def retrieve(q):
        return retriever.get_relevant_documents(q)

    all_docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(retrieve, queries)

    for doc_list in results:
        all_docs.extend(doc_list)

    return all_docs

# Load FAISS index from disk
def load_faiss_index(index_path="../data/faiss_index"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path not found: {index_path}")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Get retriever object from FAISS index
def get_retriever(index_path="../data/faiss_index", k=5):
    """Get a retriever object from the FAISS index."""
    vectorstore = load_faiss_index(index_path)
    return vectorstore.as_retriever(search_kwargs={"k": k})

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