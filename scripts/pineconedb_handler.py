from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from pinecone import Pinecone
from uuid import uuid4
import pandas as pd

class PineconeHandler:
    def __init__(self, api_key, index_name, embedding_provider='pinecone', model_name='multilingual-e5-large', namespace="default"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        
        # Initialize embeddings based on provider
        if embedding_provider == 'pinecone':
            self.embeddings = PineconeEmbeddings(model=model_name)
        else:  # openai
            self.embeddings = OpenAIEmbeddings(model=model_name)
            
        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        )
    
    def process_documents(self, data_df, text_column='text'):
        """Process DataFrame and create documents for Pinecone"""
        # Handle NaN values globally
        data_df = data_df.fillna('')
        
        # Create documents using DataFrameLoader
        loader = DataFrameLoader(
            data_df,
            page_content_column=text_column
        )
        documents = loader.load()
        
        # Generate UUIDs for documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Add documents to vectorstore
        self.vectorstore.add_documents(documents=documents, ids=uuids)
        
        doc_size = len(data_df)
        return doc_size
    
    def similarity_search(self, query, k=5):
        """Perform similarity search"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self):
        """Get retriever for use with LangChain"""
        return self.vectorstore.as_retriever()
    
    def describe_index_stats(self):
        """Get index statistics"""
        return self.vectorstore.index.describe_index_stats()