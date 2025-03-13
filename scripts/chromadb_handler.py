# chromadb_handler.py
import os
import chromadb
from tqdm import tqdm
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Import config values
from config.config import CHROMADB_PERSIST_DIR, CHROMADB_COLLECTION_NAME, EMBEDDING_MODEL_NAME

# local_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def initialize_chroma_client():
    """Instantiate and return a Chroma client."""
    client = chromadb.Client(Settings())
    return client

def get_or_create_collection(persist=CHROMADB_PERSIST_DIR, collection_name=CHROMADB_COLLECTION_NAME, embedding_model=None):
    """Return a collection, creating one if it doesn't exist."""
    if embedding_model is None:
        # Create default embedding if none provided
        model_name = "sentence-transformers/all-MiniLM-L12-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    # Create or load the collection directly using LangChain's Chroma
    vectorstore = Chroma(
        persist_directory=persist,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    return vectorstore

def create_dynamic_embedding_text(row, selected_fields, field_labels=None):
    """
    Construct an embedding text string from a DataFrame row using selected fields.
    If field_labels is provided, each field is prepended with its label.
    """
    parts = []
    for field in selected_fields:
        value = row.get(field, "")
        if field_labels and field in field_labels:
            parts.append(f"{field_labels[field]}: {value}")
        else:
            parts.append(str(value))
    return " ".join(parts)

def concatenate_fields(row, fields, separator=" "):
    """
    Concatenate the values of a list of fields from a row using the specified separator.
    """
    return separator.join(str(row.get(field, "")) for field in fields).strip()

def upsert_medicare_documents(collection, data_df, selected_fields, join_option=False, separator=" " , batch_size=10000):
    """
    Process the DataFrame and upsert documents into the Chroma collection.
    Uses dynamic field selection based on 'selected_fields'. If join_option is True,
    raw values are concatenated; otherwise, labels are added.
    Returns a list of Document objects for LangChain.
    """
    documents = []
    doc_ids = []

    # Create a basic mapping for labels (e.g., replace underscores with spaces)
    field_labels = {field: field.replace("_", " ") for field in selected_fields}

    for i, row in tqdm(data_df.iterrows(), total=data_df.shape[0], desc="Processing rows"):
        if join_option:
            embedding_text = concatenate_fields(row, selected_fields, separator)
        else:
            embedding_text = create_dynamic_embedding_text(row, selected_fields, field_labels)
        # Create a unique ID (combining NPI and row index)
        unique_id = f"{row.get('Rndrng_NPI', 'unknown')}_{i}" 
        documents.append(Document(page_content=embedding_text, metadata=row.to_dict()))
        doc_ids.append(unique_id)

    num_batches = (len(documents) // batch_size) + 1
    for batch_idx in tqdm(range(num_batches), desc="Batch Upserting"):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_ids = doc_ids[start:end]
        if batch_docs:
            collection.add_documents(documents=batch_docs, ids=batch_ids)
    return documents