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
    def __init__(self, persist_directory, collection_name, embedding_model="all-MiniLM-L6-v2"):
        """Initialize ChromaDB handler with optimized embeddings."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Get device for optimized compute
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize HuggingFace embeddings with device
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            show_progress=True
        )
        print(f"Model {embedding_model} loaded successfully on {device}")

        # you could also use psutil to calculate available memory 
        # to determine which chromadb client to use

        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        persistent_client = chromadb.PersistentClient(path=self.persist_directory) # use PersistentClient for larger datasets
        collection = persistent_client.get_or_create_collection(self.collection_name)
        
        self.vector_store = Chroma(
            client=persistent_client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
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

    def get_retriever(self, k=20):
        """Get retriever for use with LangChain.
        
        Args:
            k: Number of documents to retrieve (default: 20 for better coverage)
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )