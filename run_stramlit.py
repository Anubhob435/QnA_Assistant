import os
# No need for dotenv when using Streamlit secrets
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

# Use Streamlit secrets instead of environment variables for GitHub hosting
# No need to load_dotenv() when using Streamlit secrets

# Get config from Streamlit secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = st.secrets["PINECONE_INDEX_NAME"]
pinecone_host = st.secrets.get("PINECONE_HOST", None)

# Initialize Pinecone with API key from Streamlit secrets
pc = Pinecone(api_key=pinecone_api_key)

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
        
        # Format the source documents for display
        context_snippets = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            context_snippets.append(f"**Snippet {i+1}** (from {source}):\n{content}")
        
        context_text = "\n\n".join(context_snippets)
        
        return {
            "answer": result["output_text"],
            "context_snippets": context_text,
            "tool_used": "DocumentQA (RAG Pipeline)"
        }
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        return {
            "answer": f"An error occurred: {str(e)}",
            "context_snippets": "",
            "tool_used": "DocumentQA (Error)"
        }

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
        return {
            "answer": f"Calculation result: {result}",
            "context_snippets": f"Expression: `{clean_query}` = {result}",
            "tool_used": "Calculator Tool"
        }
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return {
            "answer": f"I couldn't perform that calculation. Error: {str(e)}",
            "context_snippets": "",
            "tool_used": "Calculator Tool (Error)"
        }

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
                    context_text = f"API Source: Dictionary API (api.dictionaryapi.dev)\nWord: {word}"
                    return {
                        "answer": f"Definition of {word}: {definition}",
                        "context_snippets": context_text,
                        "tool_used": "Dictionary Tool"
                    }
            return {
                "answer": f"I couldn't find a clear definition for '{word}'.",
                "context_snippets": "No definition data found in API response.",
                "tool_used": "Dictionary Tool"
            }
        else:
            return {
                "answer": f"I couldn't find a definition for '{word}'.",
                "context_snippets": f"API returned status code: {response.status_code}",
                "tool_used": "Dictionary Tool"
            }
    except Exception as e:
        logger.error(f"Dictionary lookup error: {e}")
        return {
            "answer": f"I couldn't look up that definition. Error: {str(e)}",
            "context_snippets": "",
            "tool_used": "Dictionary Tool (Error)"
        }

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
    
    Returns a dictionary with:
    - answer: The final answer text
    - context_snippets: The context used to generate the answer
    - tool_used: Which tool/agent was used to process the query
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
    
    # Add information about Streamlit secrets configuration
    with st.sidebar:
        if st.checkbox("Show Configuration Status", value=False):
            st.subheader("Configuration Status")
            
            # Check if secrets are available
            missing_secrets = []
            for key in ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "GEMINI_API_KEY"]:
                try:
                    if not st.secrets[key]:
                        missing_secrets.append(key)
                except:
                    missing_secrets.append(key)
                    
            if missing_secrets:
                st.error(f"Missing required secrets: {', '.join(missing_secrets)}")
            else:
                st.success("All required secrets are configured")
                
            # Check optional secrets
            optional = ["PINECONE_HOST", "PINECONE_DIMENSIONS", "EMBEDDING_MODEL_NAME"]
            for key in optional:
                try:
                    if key in st.secrets:
                        st.info(f"{key} is configured")
                except:
                    st.warning(f"{key} is not configured (optional)")
    
    # Initialize the RAG pipeline (cached)
    retriever, qa_chain = initialize_rag_pipeline()
    
    if not retriever or not qa_chain:
        st.error("Failed to initialize RAG pipeline. Please check the logs for details.")
        st.error("Make sure your Streamlit secrets are configured correctly.")
        
        # Provide more detailed guidance on fixing the issue
        with st.expander("How to fix this issue"):
            st.markdown("""
            ### Troubleshooting Steps
            
            1. Verify your Streamlit secrets are correctly configured with:
               - `PINECONE_API_KEY`
               - `PINECONE_INDEX_NAME` 
               - `GEMINI_API_KEY`
            
            2. Check that your Pinecone index exists and is accessible
            
            3. For local development, you can use a `.env` file instead
            
            4. Check the application logs for more details about the error
            """)
            
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
                
                # Display which tool/agent branch was used
                st.info(f"🔧 **Tool Used**: {response['tool_used']}")
                
                # Display the final answer
                st.markdown(f"**Answer**:\n{response['answer']}")
                
                # Display retrieved context snippets in an expandable section
                with st.expander("View Retrieved Context"):
                    st.markdown(response['context_snippets'])
        
        # Add assistant response to chat history with formatted content
        formatted_response = f"""🔧 **Tool Used**: {response['tool_used']}

**Answer**:
{response['answer']}

<details>
<summary>View Retrieved Context</summary>

{response['context_snippets']}
</details>
"""
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})# Add sidebar with app information
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