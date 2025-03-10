# config/config.py
import os
from dotenv import load_dotenv
from config_parser import parse_config

# Load environment variables (API keys, etc.)
env_path = os.path.join(os.path.dirname(__file__), '..', 'secrets.env')
load_dotenv(env_path)

# Load sensitive settings from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Load non-sensitive configuration from the JSON file.
user_config = parse_config()

# Use the JSON config if available, otherwise default to env values or hardcoded defaults.
CHROMADB_SERVER_HOST = user_config.get("CHROMADB_SERVER_HOST", os.getenv("CHROMADB_SERVER_HOST", "localhost"))
CHROMADB_SERVER_PORT = user_config.get("CHROMADB_SERVER_PORT", os.getenv("CHROMADB_SERVER_PORT", "8000"))
CHROMADB_COLLECTION_NAME = user_config.get("CHROMADB_COLLECTION_NAME", os.getenv("CHROMADB_COLLECTION_NAME", "example_collection"))
PERSIST_DIR = user_config.get("PERSIST_DIR", os.getenv("PERSIST_DIR", "./chroma_db"))
EMBEDDING_MODEL_NAME = user_config.get("EMBEDDING_MODEL_NAME", os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002"))
MEDICARE_DATA_PATH = user_config.get("MEDICARE_DATA_PATH", os.getenv("MEDICARE_DATA_PATH", "./data/medicare.csv"))
LOGGING_LEVEL = user_config.get("LOGGING_LEVEL", os.getenv("LOGGING_LEVEL", "INFO"))
