# chromadb_handler.py
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from uuid import uuid4

class ChromaDBHandler:
    def __init__(self, persist_directory, collection_name, embedding_model="all-MiniLM-L6-v2"):
        """Initialize ChromaDB handler with HuggingFace embeddings."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
            
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