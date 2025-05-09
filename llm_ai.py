import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_llm():
    """
    Initialize and return a Gemini 2.0 Flash language model.
    API key is loaded from Streamlit secrets when deployed,
    or from environment variables during local development.
    """
    try:
        # First try to get API key from Streamlit secrets (for deployment)
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        print("Using Gemini API key from Streamlit secrets")
    except (KeyError, RuntimeError) as e:
        # Fall back to environment variables (for local development)
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in Streamlit secrets or environment variables") from e
        print("Using Gemini API key from environment variables")
    
    # Initialize the Gemini model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            google_api_key=gemini_api_key
        )
        return llm
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        raise
    
    return llm