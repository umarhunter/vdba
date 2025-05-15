import pandas as pd

from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from tqdm import tqdm

class PGVectorHandler:
    def __init__(self, connection_string, collection_name, embedding_model="all-MiniLM-L6-v2"):
        """Initialize PGVector handler with HuggingFace embeddings."""
        self.connection_string = connection_string
        self.collection_name = collection_name
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
            
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True
        )

    def process_documents(self, data_df, text_column='text', batch_size=100):
        """Process DataFrame and add documents to PGVector with batching."""

        # Add id column based on index
        data_df['id'] = data_df.index.astype(str)  # Ensure ID is also string type
        data_df = data_df.fillna('')  # Replace all NaN values with empty strings

        # Create documents using DataFrameLoader
        loader = DataFrameLoader(data_df, page_content_column=text_column)
        documents = loader.load()
        
        # Generate all UUIDs upfront
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # after doing some research it appears that there is a psycopg3 driver limit to parameters (65,535) so we need to batch the inserts
        batch_size = 100  # Adjust as needed

        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            
        doc_size = len(data_df)
        return doc_size

    def similarity_search(self, query, k=5):
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self):
        """Get retriever for use with LangChain."""
        return self.vector_store.as_retriever()