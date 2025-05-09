import streamlit as st

st.title("Testing Secrets")

if "PINECONE_API_KEY" in st.secrets:
    st.success("✅ PINECONE_API_KEY found")
else:
    st.error("❌ PINECONE_API_KEY not found")

if "GEMINI_API_KEY" in st.secrets:
    st.success("✅ GEMINI_API_KEY found")
else:
    st.error("❌ GEMINI_API_KEY not found")

# Check all required secrets
required_secrets = [
    "PINECONE_API_KEY", 
    "PINECONE_INDEX_NAME", 
    "PINECONE_HOST", 
    "PINECONE_DIMENSIONS", 
    "EMBEDDING_MODEL_NAME",
    "GEMINI_API_KEY"
]

missing = []
for secret in required_secrets:
    if secret not in st.secrets:
        missing.append(secret)
        
if missing:
    st.error(f"Missing secrets: {', '.join(missing)}")
else:
    st.success("All required secrets are properly configured")
