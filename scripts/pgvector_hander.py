import pandas as pd
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4

class PGVectorHandler:
    """
    A class to handle PGVector operations including document processing and similarity search.
    """
    def __init__(self, connection_string, collection_name, embedding_model="text-embedding-ada-002"):
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vectorstore = PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def process_documents(self, data_df, text_column='text', max_docs=16000):
        """Process DataFrame and add documents to PGVector. Limit to max_docs."""
        data_df = data_df.fillna('')
        if len(data_df) > max_docs:
            raise ValueError(f"PGVector (Docker) supports up to {max_docs} documents. Provided: {len(data_df)}")
        loader = DataFrameLoader(data_df, page_content_column=text_column)
        documents = loader.load()
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)
        return documents

    def similarity_search(self, query, k=5):
        """Perform similarity search."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self):
        """Get retriever for use with LangChain."""
        return self.vectorstore.as_retriever()