# chromadb_handler.py
import chromadb
import torch
import os
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4

class ChromaDBHandler:
    def __init__(self, persist_directory, collection_name, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize ChromaDB handler with GPU-enabled embeddings if available."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure the model is downloaded
        try:
            # This will download the model if it's not already cached
            _ = SentenceTransformer(embedding_model)
            print(f"Successfully loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to a simpler model if available
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"Falling back to default model: {embedding_model}")
        
        # Detect best available device
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU for ChromaDB: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon GPU for ChromaDB")
        else:
            device = "cpu"
            print("Using CPU for ChromaDB embeddings")

        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': False},
            cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
    def process_documents(self, data_df, text_column='text'):
        """Process DataFrame and add documents to ChromaDB."""
        # Convert text list to Document objects with IDs
        documents = [
            Document(
                page_content=text,
                metadata={'id': str(uuid4())}
            ) for text in data_df[text_column].tolist()
        ]
        
        # Add documents to the collection
        self.vector_store.add_documents(documents=documents)
        return len(documents)

    def similarity_search(self, query, k=5):
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self):
        """Get retriever for use with LangChain."""
        return self.vector_store.as_retriever()