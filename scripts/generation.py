from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval import retrieve_chunks, get_retriever, retrieve_multiple_queries
from config import gemini_api_key, google_api_key
from query_expansion import expand_query

# Format context from chunks into a prompt
def format_context(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text

# Generate answer using Gemini
def generate_answer(query, retriever=None, expand=True):
    if retriever is None:
        retriever = get_retriever()

    if expand:
        reformulations = expand_query(query)
        all_queries = [query] + reformulations
        print("\n🔍 Expanded Queries:")
        for i, q in enumerate(all_queries, 1):
            print(f"{i}. {q}")
    else:
        all_queries = [query]

    # Retrieve documents for all queries
    all_docs = retrieve_multiple_queries(all_queries, retriever)

    # Remove duplicate docs by content hash
    seen = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)

    # Construct context string
    context = "\n---\n".join([doc.page_content for doc in unique_docs])

    # Prompt Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    final_prompt = (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )
    response = llm.invoke(final_prompt)
    return response.content.strip()

# Optional: direct CLI test
if __name__ == "__main__":
    user_query = input("Ask your question: ")
    answer = generate_answer(user_query)
    print("\n--- Gemini's Answer ---\n")
    print(answer)