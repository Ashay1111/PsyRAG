from retrieval import get_retriever
from generation import generate_answer
from config import gemini_api_key

retriever = get_retriever()
query = "What is prospect theory and how is it different from classical economics?"
answer = generate_answer(query, retriever=retriever)
print("\nGemini's answer:\n", answer)