# QnA Assistant

A powerful question-answering system built with Retrieval-Augmented Generation (RAG) that leverages the Gemini 2.0 Flash LLM and Pinecone vector database for enhanced information retrieval.

#### Live Demo

The application is also available online at: [https://qnaassistant-agent.streamlit.app/](https://qnaassistant-agent.streamlit.app/)

Sample documents can be found in the GitHub repository:
[https://github.com/Anubhob435/QnA_Assistant/tree/master/data](https://github.com/Anubhob435/QnA_Assistant/tree/master/data)

## 📝 Overview

This QnA Assistant is designed to answer questions based on a collection of document data stored in a vector database. The system uses a RAG approach, retrieving relevant document chunks before generating accurate answers. It features:

- **Multi-agent Architecture**: Uses a routing agent to direct queries to the appropriate tool
- **Document Retrieval**: Fetches relevant document chunks from Pinecone vector database
- **LLM Integration**: Employs Google's Gemini 2.0 Flash model for answer generation
- **Multiple Interfaces**: Available through both a CLI and Streamlit web interface
- **Basic Calculator**: Built-in calculator tool for performing mathematical operations
- **Define**: Serves as General Dictionary

## 🔧 Technology Stack

- **LLM**: Google's Gemini 2.0 Flash
- **Vector Database**: Pinecone
- **Embeddings**: BAAI/llama-2-embeddings (with fallback to all-MiniLM-L6-v2)
- **Framework**: LangChain for agent and chain orchestration
- **Web UI**: Streamlit
- **Logging**: Python's logging module

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pinecone API key
- Google Generative AI (Gemini) API key

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

#### For Local Development
Create a `.env` file in the project root with the following variables:
```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_HOST=your_pinecone_host_url  # Optional
PINECONE_DIMENSIONS=768  # For BAAI/llama-2-embeddings
EMBEDDING_MODEL_NAME=BAAI/llama-2-embeddings
GEMINI_API_KEY=your_gemini_api_key
```

#### For Streamlit Cloud Deployment
When deploying to Streamlit Cloud, create a `.streamlit/secrets.toml` file:
```toml
PINECONE_API_KEY = "your_pinecone_api_key"
PINECONE_INDEX_NAME = "your_index_name"
PINECONE_HOST = "your_pinecone_host_url"  # Optional
PINECONE_DIMENSIONS = "768"  # For BAAI/llama-2-embeddings
EMBEDDING_MODEL_NAME = "BAAI/llama-2-embeddings"
GEMINI_API_KEY = "your_gemini_api_key"
```

> **Note:** Make sure to add `.streamlit/secrets.toml` to your `.gitignore` file to prevent committing sensitive API keys to GitHub.

### Data Preparation

Place your text files in the `/data` directory. The system currently includes sample files:
- FAQ_general.txt
- FAQ_product.txt
- functional_requirements.txt
- product_specifications.txt
- technical_specifications.txt

Sample documents can be found in the GitHub repository:
[https://github.com/Anubhob435/QnA_Assistant/tree/master/data](https://github.com/Anubhob435/QnA_Assistant/tree/master/data)

## 📊 Usage

### Initial Setup

Run the main script to process documents and create the vector index:

```bash
python main.py
```

### Command Line Interface

Launch the CLI version:

```bash
python run_CLI.py
```

### Web Interface

Launch the Streamlit web app:

```bash
python run_stramlit.py
```

Or use the main script with the web flag:

```bash
python main.py --web
```

## 🧩 System Components

### Document Processing

- Documents are loaded from the data directory
- Text is split into chunks using LangChain's CharacterTextSplitter
- Document chunks are embedded and stored in Pinecone

### Query Processing

1. User submits a question
2. Agent determines whether to use the RAG pipeline or calculator
3. For RAG queries:
   - Relevant document chunks are retrieved from Pinecone
   - Retrieved context and query are passed to the LLM
   - LLM generates an answer based on the context
4. Results are displayed to the user

### Logging

All operations are logged with timestamps in the `agent_logs` directory.

## 🛠️ Project Structure

```
QnA_Assistant/
├── data/                    # Document storage
│   ├── FAQ_general.txt
│   ├── FAQ_product.txt
│   ├── functional_requirements.txt
│   ├── product_specifications.txt
│   └── technical_specifications.txt
├── agent_logs/              # Log file storage
├── main.py                  # Main application (document processing + interface)
├── llm_ai.py                # LLM configuration (Gemini)
├── run_CLI.py               # CLI-only version (assumes documents are processed)
├── run_stramlit.py          # Streamlit-only version (assumes documents are processed)
├── requirements.txt         # Project dependencies
├── actions.txt              # Development steps/checklist
└── README.md                # This file
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the agent framework
- Google for the Gemini 2.0 Flash LLM
- Pinecone for vector database hosting

## 💡 Future Enhancements

- Add support for more data sources (PDFs, websites)
- Implement memory for multi-turn conversations
- Add authentication system for the web interface
- Implement additional tools beyond RAG and calculator
- Support for document updates without full reprocessing

## 📄 License
MIT License – Free to use, modify, and distribute.

## 👨‍💻 Author
ANUBHOB DEY
AI Engineer & Backend Developer