from config import gemini_api_key, google_api_key
from query_expansion import expand_query

query = "What is prospect theory and how does it challenge classical economic thinking?"
expansions = expand_query(query)

for q in expansions:
    print(f"{q}")
