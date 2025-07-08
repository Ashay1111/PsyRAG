from retrieval import get_retriever
from generation import generate_answer

retriever = get_retriever()
query = input("Ask a question about psychology, cognition, or behavior: ")
answer = generate_answer(query, retriever=retriever)
print("\nPsyRAG:\n", answer)