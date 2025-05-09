import pandas as pd
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from tqdm import tqdm

class PGVectorHandler:
    def __init__(self, connection_string, collection_name, embedding_model="text-embedding-ada-002"):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def process_documents(self, data_df, text_column='text', batch_size=100):
        """Process DataFrame and add documents to PGVector with batching."""
        # Handle NaN values
        data_df = data_df.fillna('')
        
        # Create documents using DataFrameLoader
        loader = DataFrameLoader(data_df, page_content_column=text_column)
        documents = loader.load()
        
        # Generate UUIDs for all documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Process in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = uuids[i:i + batch_size]
            
            try:
                self.vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                continue
        
        return documents

    def similarity_search(self, query, k=5):
        """Perform similarity search."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self):
        """Get retriever for use with LangChain."""
        return self.vectorstore.as_retriever()