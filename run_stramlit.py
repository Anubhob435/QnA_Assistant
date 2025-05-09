import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging
import datetime
import streamlit as st
from llm_ai import get_gemini_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"agent_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 RAG-Powered Assistant")
    st.write("Ask questions about the documents that have been uploaded to Pinecone.")
    
    # Initialize the RAG pipeline (cached)
    retriever, qa_chain = initialize_rag_pipeline()
    
    if not retriever or not qa_chain:
        st.error("Failed to initialize RAG pipeline. Please check the logs for details.")
        return
    
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
                response = ask_question(prompt, retriever, qa_chain)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add sidebar with app information
    with st.sidebar:
        st.header("About")
        st.write("This RAG (Retrieval-Augmented Generation) Assistant uses:")
        st.write("- Gemini 2.0 Flash LLM")
        st.write("- Pinecone Vector Database")
        st.write("- LangChain Framework")
        
        # Add option to clear chat history
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

# Run the Streamlit app
if __name__ == "__main__":
    logger.info("Starting Streamlit app")
    main()