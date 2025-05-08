import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from tqdm.auto import tqdm

# Load configuration from .env file
load_dotenv()

# Initialize Pinecone with API key from .env
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Get index information from environment variables
index_name = os.getenv("PINECONE_INDEX_NAME")
host = os.getenv("PINECONE_HOST")
dimensions = int(os.getenv("PINECONE_DIMENSIONS"))
model_name = os.getenv("EMBEDDING_MODEL_NAME")

# Check if index exists or create it
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    index_description = pc.describe_index(name=index_name)
    if index_description.dimension != dimensions:
        print(f"Deleting existing Pinecone index '{index_name}' as its dimension ({index_description.dimension}) does not match the required dimension ({dimensions})...")
        pc.delete_index(index_name)
        print(f"Creating new Pinecone index '{index_name}' with dimension {dimensions}...")
        pc.create_index(
            name=index_name,
            dimension=dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new Pinecone index '{index_name}'")
    else:
        print(f"Using existing Pinecone index '{index_name}'")
elif index_name not in existing_indexes:
    print(f"Creating new Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimensions,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created new Pinecone index '{index_name}'")

# Get a reference to the index
index = pc.Index(index_name)

# Load all text files from the data directory
data_dir = "data"
documents = []
for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        print(f"Loading {file_path}...")
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} document chunks")

try:
    # First try using a community version of the Llama text embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/llama-2-embeddings",  # Using a compatible embedding model
    )
    print("Using BAAI/llama-2-embeddings embedding model")
except Exception as e:
    print(f"Failed to load Llama embedding model: {e}")
    print("Falling back to all-MiniLM-L6-v2 embedding model")
    # If unable to load Llama model, fall back to all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Pinecone vector store
print("Uploading documents to Pinecone...")
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f"Successfully uploaded {len(chunks)} document chunks to Pinecone index '{index_name}'")
