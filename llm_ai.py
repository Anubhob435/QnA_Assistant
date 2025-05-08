import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

def get_gemini_llm():
    """
    Initialize and return a Gemini 2.0 Flash language model.
    """
    # The API key is loaded from the .env file
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key=gemini_api_key
    )
    
    return llm