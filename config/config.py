# config/config.py
import os
from dotenv import load_dotenv
import os.path

# Load environment variables from the secrets.env file (located in the project root)
env_path = os.path.join(os.path.dirname(__file__), '..', 'secrets.env')
load_dotenv(env_path)

# Sensitive configuration (API keys)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ChromaDB settings
CHROMADB_PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")
CHROMADB_COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION_NAME", "new_york_medicare")

# Embedding model name (for HuggingFaceEmbeddings)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L12-v2")

# Medicare dataset path
MEDICARE_DATA_PATH = os.getenv("MEDICARE_DATA_PATH", "./data/insurance/medicare/2022/sample_ny_data.csv")

# Default model choice for LLM chatbot ("deepseek", "llama2", or "openai")
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "deepseek")

# Other configuration as needed
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
