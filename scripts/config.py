import os
from dotenv import load_dotenv

load_dotenv() 
gemini_api_key = os.getenv("GEMINI_API_KEY")
google_api_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")