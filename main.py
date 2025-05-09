import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec  # Updated import
import logging
import datetime
from llm_ai import get_gemini_llm  # Import the Gemini LLM function

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

# Load configuration from .env file
load_dotenv()

# Initialize Pinecone with API key from .env (new method)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Get index information from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME")
dimensions = int(os.getenv("PINECONE_DIMENSIONS"))
model_name = os.getenv("EMBEDDING_MODEL_NAME")

# Check if index exists or create it
try:
    # Get list of index names
    index_list = pc.list_indexes().names()
    
    if index_name in index_list:
        logger.info(f"Using existing Pinecone index '{index_name}'")
    else:
        logger.info(f"Creating new Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info(f"Created new Pinecone index '{index_name}'")
except Exception as e:
    logger.error(f"Error with Pinecone index: {e}")
    raise

# Get the index instance
index = pc.Index(index_name)

# Load all text files from the data directory
data_dir = "data"
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        logger.info(f"Loading {file_path}...")
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
logger.info(f"Created {len(chunks)} document chunks")

try:
    # First try using a community version of the Llama text embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/llama-2-embeddings",  # Using a compatible embedding model
    )
    logger.info("Using BAAI/llama-2-embeddings embedding model")
except Exception as e:
    logger.error(f"Failed to load Llama embedding model: {e}")
    logger.info("Falling back to all-MiniLM-L6-v2 embedding model")
    # If unable to load Llama model, fall back to all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Pinecone vector store
logger.info("Uploading documents to Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

logger.info(f"Successfully uploaded {len(chunks)} document chunks to Pinecone index '{index_name}'")

# RAG pipeline
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Initialize the Gemini LLM
llm = get_gemini_llm()
logger.info("Initialized Gemini 2.0 Flash model")

# Set up retriever and QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

def run_rag_pipeline(query):
    """
    Run the Retrieval-Augmented Generation pipeline
    """
    logger.info(f"RAG Query: {query}")
    docs = retriever.get_relevant_documents(query)
    logger.info(f"Retrieved {len(docs)} relevant documents")
    return qa_chain({"input_documents": docs, "question": query})

# Agentic workflow
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StdOutCallbackHandler

# Custom callback handler to log agent steps
class LoggingCallbackHandler(StdOutCallbackHandler):
    def on_agent_action(self, action, **kwargs):
        """Log agent action decision"""
        logger.info(f"Agent decided to use: {action.tool}")
        logger.info(f"Tool input: {action.tool_input}")
        
    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish"""
        logger.info(f"Agent finished: {finish.return_values}")

# Create a simple calculator tool
def simple_calculator(query):
    """Simple calculator function"""
    try:
        logger.info(f"Calculator input: {query}")
        result = str(eval(query))
        logger.info(f"Calculator result: {result}")
        return result
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return "Calculation error."

# Define tools
tools = [
    Tool(
        name="Calculator", 
        func=simple_calculator, 
        description="For math/calculation tasks, input should be a mathematical expression."
    ),
    Tool(
        name="RAG_QA", 
        func=run_rag_pipeline, 
        description="For general Q&A from documents. Use this for most questions."
    )
]

# Initialize agent with callback handler for logging
callback_handler = LoggingCallbackHandler()
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    callbacks=[callback_handler]
)

# CLI demo
def run_cli_demo():
    logger.info("Starting CLI demo")
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        logger.info(f"User query: {query}")
        try:
            response = agent.run(query)
            print("\n" + response)
        except Exception as e:
            logger.error(f"Error during agent execution: {e}")
            print(f"\nError: {str(e)}")

# Streamlit app
def run_streamlit_app():
    import streamlit as st

    st.title("📘 RAG-Powered Multi-Agent Assistant")
    st.write("Ask questions about the loaded documents or perform calculations.")

    query = st.text_input("Enter your question:")
    if query:
        try:
            st.info("Processing your question... Please wait.")
            response = agent.run(query)
            st.write("### 💬 Answer:")
            st.write(response)
            
            # Show the logs in an expander for transparency
            with st.expander("View Agent Process"):
                # This would ideally show logs, but for now we'll just add a placeholder
                st.text("Agent process logs would appear here in a real deployment")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Main execution
if __name__ == "__main__":
    import sys
    
    logger.info("QnA Assistant starting up")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        logger.info("Starting in web mode")
        run_streamlit_app()
    else:
        run_cli_demo()

