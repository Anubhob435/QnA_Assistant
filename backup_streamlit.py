import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging
import datetime
import streamlit as st
from llm_ai import get_gemini_llm
import re
import requests
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# Ensure agent_logs directory exists
os.makedirs("agent_logs", exist_ok=True)

# Configure logging with log files in agent_logs folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"agent_logs/stream_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pinecone with API key from .env
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Get index information from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def initialize_rag_pipeline():
    """
    Initialize the RAG pipeline components
    This is cached to prevent reinitializing on every Streamlit rerun
    """
    logger.info("Initializing RAG pipeline components")
    
    try:
        # Get the Pinecone index instance
        index = pc.Index(index_name)
        
        # Initialize embedding model
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/llama-2-embeddings",
            )
            logger.info("Using BAAI/llama-2-embeddings embedding model")
        except Exception as e:
            logger.error(f"Failed to load Llama embedding model: {e}")
            logger.info("Falling back to all-MiniLM-L6-v2 embedding model")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Connect to existing vector store (no document upload)
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
        )
        logger.info(f"Connected to existing Pinecone index '{index_name}'")

        # Initialize the Gemini LLM
        llm = get_gemini_llm()
        logger.info("Initialized Gemini 2.0 Flash model")

        # Set up RAG pipeline
        from langchain.chains.qa_with_sources import load_qa_with_sources_chain

        # Set up retriever and QA chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        
        return retriever, qa_chain
    
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        st.error(f"Error initializing RAG pipeline: {e}")
        return None, None

def ask_question(query, retriever, qa_chain):
    """
    Run a query through the RAG pipeline
    """
    logger.info(f"Query: {query}")
    try:
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} relevant documents")
        result = qa_chain({"input_documents": docs, "question": query})
        return result["output_text"]
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return f"An error occurred: {str(e)}"

# Helper tools for the agent
def simple_calculator(query):
    """Simple calculator function that evaluates mathematical expressions"""
    try:
        # Clean the query to get just the mathematical expression
        # Remove words like calculate, compute, etc.
        clean_query = re.sub(r'(calculate|compute|evaluate|what is|find)', '', query, flags=re.IGNORECASE).strip()
        logger.info(f"Calculator input: {clean_query}")
        result = eval(clean_query)
        logger.info(f"Calculator result: {result}")
        return f"Calculation result: {result}"
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return f"I couldn't perform that calculation. Error: {str(e)}"

def dictionary_lookup(query):
    """Dictionary tool to define words"""
    try:
        # Extract the word to define
        match = re.search(r'(define|meaning of|definition of|what is|what does) ([a-zA-Z]+)', query, flags=re.IGNORECASE)
        if match:
            word = match.group(2).lower().strip()
        else:
            word = query.strip()
            
        logger.info(f"Dictionary lookup for: {word}")
        
        # Use the Free Dictionary API
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                meanings = data[0].get('meanings', [])
                if meanings:
                    definition = meanings[0].get('definitions', [{}])[0].get('definition', 'No definition found.')
                    return f"Definition of {word}: {definition}"
            return f"I couldn't find a clear definition for '{word}'."
        else:
            return f"I couldn't find a definition for '{word}'."
    except Exception as e:
        logger.error(f"Dictionary lookup error: {e}")
        return f"I couldn't look up that definition. Error: {str(e)}"

def route_to_rag(query, retriever, qa_chain):
    """Router to the RAG pipeline"""
    logger.info(f"Routing to RAG: {query}")
    return ask_question(query, retriever, qa_chain)

@st.cache_resource
def initialize_agent_framework(_llm):
    """Initialize the agent framework with tools"""
    logger.info("Initializing agent framework")
    
    # Define tools that will use the retriever and qa_chain
    # These will be initialized later when processing queries
    tools = [
        Tool(
            name="Calculator", 
            func=simple_calculator, 
            description="For math/calculation tasks. Use when the query contains 'calculate', 'compute', or is a mathematical expression."
        ),
        Tool(
            name="Dictionary", 
            func=dictionary_lookup, 
            description="For definition lookups. Use when the query contains 'define', 'definition', 'meaning of', etc."
        ),
        Tool(
            name="DocumentQA", 
            func=lambda query: "Use the RAG pipeline for document questions", 
            description="For general questions about documents. Use this for most questions."
        )
    ]
    
    # Initialize agent
    agent = initialize_agent(
        tools, 
        _llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    
    return agent

def process_query(query, retriever, qa_chain, agent):
    """
    Process user query through the agentic workflow
    
    If the query contains keywords like "calculate" or "define", route to appropriate tool
    Otherwise, use the RAG pipeline
    """
    logger.info(f"Processing query: {query}")
    
    # Direct routing based on keywords (faster than letting the agent decide every time)
    if re.search(r'\b(calculate|compute|evaluate|what is \d|find the value|solve)\b', query, flags=re.IGNORECASE):
        logger.info("Detected calculation request, routing directly to calculator tool")
        return simple_calculator(query)
    
    elif re.search(r'\b(define|meaning of|definition of|what does .* mean)\b', query, flags=re.IGNORECASE):
        logger.info("Detected definition request, routing directly to dictionary tool")
        return dictionary_lookup(query)
    
    else:
        # For more complex decisions, use the RAG pipeline
        logger.info("No specialized tool pattern detected, routing to RAG")
        return route_to_rag(query, retriever, qa_chain)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Agentic RAG Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 Agentic RAG Assistant")
    st.write("Ask questions about the documents, request calculations, or ask for definitions.")
    
    # Initialize the RAG pipeline (cached)
    retriever, qa_chain = initialize_rag_pipeline()
    
    if not retriever or not qa_chain:
        st.error("Failed to initialize RAG pipeline. Please check the logs for details.")
        return
    
    # Initialize the LLM
    llm = get_gemini_llm()
    
    # Initialize the agent framework
    agent = initialize_agent_framework(llm)
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process the query through our agentic workflow
                response = process_query(prompt, retriever, qa_chain, agent)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add sidebar with app information
    with st.sidebar:
        st.header("About")
        st.write("This Agentic RAG Assistant uses:")
        st.write("- Gemini 2.0 Flash LLM")
        st.write("- Pinecone Vector Database")
        st.write("- LangChain Framework")
        st.write("- Calculator & Dictionary tools")
        
        # Add option to clear chat history
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    logger.info("Starting Streamlit app")
    main()