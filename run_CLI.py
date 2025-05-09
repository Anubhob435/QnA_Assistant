import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging
import datetime
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

# Get the index instance
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

def ask_question(query):
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

# CLI demo
def run_cli():
    logger.info("Starting CLI interface")
    print("\n===== RAG Assistant CLI =====")
    print("Documents have already been uploaded to Pinecone.")
    print("Ask questions about the loaded documents.")
    
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        logger.info(f"User query: {query}")
        try:
            response = ask_question(query)
            print("\n" + response)
        except Exception as e:
            logger.error(f"Error during query execution: {e}")
            print(f"\nError: {str(e)}")

# Main execution
if __name__ == "__main__":
    logger.info("RAG Assistant starting up")
    run_cli()

