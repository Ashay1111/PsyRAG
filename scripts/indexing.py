# %%
import os
from langchain.document_loaders import PyMuPDFLoader, PDFMinerLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema.document import Document
from dotenv import load_dotenv
from config import gemini_api_key, google_api_key



# # %%
# # Set up environment variables
# load_dotenv()  # Loads variables from .env into environment
# os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Ensure this is set to your service account key path
# api_key = os.getenv("GEMINI_API_KEY")


# %%
def load_files_from_folder(folder_path: str, source_type: str) -> list[Document]:
    docs = []
    print(f"Loading PDFs from folder: {folder_path}")

    for file in sorted(os.listdir(folder_path)):
        if not file.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file)
        print(f"Processing file: {file}")

        try:
            loader = PyMuPDFLoader(file_path)
            loaded_docs = loader.load()
        except Exception as e:
            print(f"PyMuPDF failed for {file}: {e}. Trying PDFMiner.")
            try:
                loader = PDFMinerLoader(file_path)
                loaded_docs = loader.load()
            except Exception as fallback_error:
                print(f"Skipping {file}. All loaders failed: {fallback_error}")
                continue

        for doc in loaded_docs:
            doc.metadata.update({
                "source_type": source_type,
                "filename": file
            })

        docs.extend(loaded_docs)

    print(f"Total documents loaded from '{source_type}': {len(docs)}")
    return docs




# %%
def chunk_documents(docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} total chunks")
    return chunks



# %%
def embed_and_store(chunks, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Embedding documents using Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"Index saved to: {save_path}")




# %%
def main():
    books = load_files_from_folder("../books", source_type="book")
    papers = load_files_from_folder("research_papers", source_type="paper")

    all_docs = books + papers
    chunks = chunk_documents(all_docs)
    embed_and_store(chunks, save_path="../data/faiss_index")


if __name__ == "__main__":
    main()


