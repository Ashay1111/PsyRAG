from langchain_google_genai import ChatGoogleGenerativeAI


def expand_query(original_query: str, model_name: str = "gemini-2.0-flash") -> list[str]:
    """Expands a user query into multiple semantically equivalent questions."""
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    prompt = (
        f"return the following user question in three different ways without additional lines strictly. "
        f"Each reformulation should aim to capture the same core intent but use different wording or focus:\n\n"
        f"Original question: {original_query}"
    )

    response = llm.invoke(prompt)
    # Split into lines and clean up
    variations = [
        line.strip("-• ").strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    return variations[:3]  # just in case Gemini goes wild
