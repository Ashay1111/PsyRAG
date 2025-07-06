from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval import retrieve_chunks
from config import gemini_api_key, google_api_key

# Format context from chunks into a prompt
def format_context(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


# Generate answer using Gemini
def generate_answer(query: str, k: int = 5) -> str:
    # Retrieve relevant chunks
    docs = retrieve_chunks(query, k=k)
    if not docs:
        return "No relevant documents found."

    # Format context
    context = format_context(docs)

    # Build prompt
    prompt = f"""You are a brilliant psycologist. Answer the question using the context below.

Context:
{context}

Question:
{query}

Answer:"""

    # Call Gemini
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        max_retries=2,
    )
    response = model.invoke(prompt)
    return response.content


# Optional: direct CLI test
if __name__ == "__main__":
    user_query = input("Ask your question: ")
    answer = generate_answer(user_query)
    print("\n--- Gemini's Answer ---\n")
    print(answer)
