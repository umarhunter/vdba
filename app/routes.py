# routes.py
# The routes.py file contains the route definitions for the Flask application. It defines the behavior of the application when a user accesses different URLs or endpoints.

import os
import getpass
from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from scripts.chromadb_handler import (
    initialize_chroma_client,
    get_or_create_collection,
    upsert_medicare_documents,
    create_dynamic_embedding_text,
    concatenate_fields
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from scripts.data_loader import load_medicare_data
from config.config import CHROMADB_COLLECTION_NAME, CHROMADB_PERSIST_DIR, EMBEDDING_MODEL_NAME, MODEL_CHOICE

# Define the USER_CONFIG global variable with default settings.
USER_CONFIG = {
    'dataset': 'Medicare',
    'vector_db': 'ChromaDB',
    'embedding_model': 'all-mini',
    'llm': 'DeepSeek',
    'openai_api_token': '',
    'huggingface_api_token': '',
    # ChromaDB specific settings
    'chroma_settings': {
        'collection_name': 'new_york_medicare',
        'persist_directory': './chroma_db',
        'collection_metadata': {'description': 'Medicare provider data'},
        'embedding_function': 'sentence-transformers/all-MiniLM-L12-v2',
    },
    'pinecone_settings': {
        'index_name': '',
        'api_key': '',
    }
}

main = Blueprint("main", __name__)

# Initialize the ChromaDB client and embedding function.
client = initialize_chroma_client()

model_name = "sentence-transformers/all-MiniLM-L12-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
collection = get_or_create_collection(CHROMADB_PERSIST_DIR, CHROMADB_COLLECTION_NAME, embedding_model)

# Route: Home page
@main.route("/")
def index():
    return render_template("welcome.html")

# New Dashboard Route
@main.route("/dashboard")
def dashboard():
    # Render the renamed dashboard.html template.
    return render_template("dashboard.html", query="", answer="", contexts="")

# Settings Page
@main.route("/settings", methods=["GET", "POST"])
def settings():
    global USER_CONFIG
    if request.method == "POST":
        # Update basic settings
        USER_CONFIG['dataset'] = request.form.get("dataset", "Medicare")
        USER_CONFIG['vector_db'] = request.form.get("vector_db", "ChromaDB")
        USER_CONFIG['embedding_model'] = request.form.get("embedding_model", "all-mini")
        USER_CONFIG['llm'] = request.form.get("llm", "DeepSeek")
        USER_CONFIG['openai_api_token'] = request.form.get("openai_api_token", "")
        USER_CONFIG['huggingface_api_token'] = request.form.get("huggingface_api_token", "")
        
        # Update ChromaDB specific settings if ChromaDB is selected
        if USER_CONFIG['vector_db'] == 'ChromaDB':
            USER_CONFIG['chroma_settings'].update({
                'collection_name': request.form.get("chroma_collection_name", "new_york_medicare"),
                'persist_directory': request.form.get("chroma_persist_dir", "./chroma_db"),
                'collection_metadata': {
                    'description': request.form.get("chroma_description", "Medicare provider data")
                }
            })
            
            # Reinitialize ChromaDB collection with new settings
            global collection
            collection = get_or_create_collection(
                persist=USER_CONFIG['chroma_settings']['persist_directory'],
                collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                embedding_model=embedding_model
            )
        
        # Update Pinecone settings if PineconeDB is selected
        if USER_CONFIG['vector_db'] == 'PineconeDB':
            USER_CONFIG['pinecone_settings'].update({
                'index_name': request.form.get("pinecone_index", ""),
                'api_key': request.form.get("pinecone_api_key", "")
            })
        
        return redirect(url_for("main.dashboard"))
    else:
        return render_template("settings.html", config=USER_CONFIG)

# Route: Configure Embeddings
@main.route("/configure_embeddings", methods=["GET", "POST"])
def configure_embeddings():
    if request.method == "POST":
        # Get form data
        selected_fields = request.form.getlist("selected_fields")
        join_option = request.form.get("join_option") == "true"
        separator = request.form.get("separator", " ")
        
        # Load dataset based on the user selection.
        if USER_CONFIG['dataset'] == "Medicare":
            data_df = load_medicare_data()
        else:
            # Placeholder if additional datasets are added later.
            return render_template("configure_embeddings_result.html",
                                   error="Selected dataset not implemented yet.",
                                   fields=[])
        
        if not selected_fields:
            return render_template("configure_embeddings_result.html",
                                   error="Please select at least one field to embed.",
                                   fields=[])
        else:
            print("Selected fields:", selected_fields)
        
        # Process documents using the vector store
        docs = upsert_medicare_documents(
            collection=get_or_create_collection(CHROMADB_PERSIST_DIR, CHROMADB_COLLECTION_NAME, embedding_model),
            data_df=data_df,
            selected_fields=selected_fields,
            join_option=join_option,
            separator=separator
        )
        return render_template("configure_embeddings_result.html",
                               count=len(docs),
                               fields=selected_fields)
    
    # Handle GET request: show the available fields for the selected dataset.
    if USER_CONFIG['dataset'] == "Medicare":
        data_df = load_medicare_data()
        available_fields = data_df.columns.tolist()
    else:
        available_fields = []
    return render_template("configure_embeddings.html", available_fields=available_fields)


# Route: Query Documents
@main.route("/query", methods=["GET"])
def query_docs():
    query_text = request.args.get("q", "")
    if query_text:
        local_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        coll = get_or_create_collection(CHROMADB_PERSIST_DIR, CHROMADB_COLLECTION_NAME, local_embedding)
        results = coll.similarity_search(query=query_text,k=1)
        return render_template("query.html", query=query_text, results=results)
    else:
        # Render the query form if no query parameter is provided.
        return render_template("query.html", query="", results=None)

# Route: Chat with LLM
@main.route("/chat", methods=["GET"])
def chat():
    query = request.args.get("q", "")
    current_llm = USER_CONFIG['llm']
    if query:
        selected_llm = USER_CONFIG['llm'].lower()
        if selected_llm == "deepseek":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="deepseek-r1", temperature=0.0)
        elif selected_llm == "llama2":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="llama2", temperature=0.0)
        elif selected_llm == "openai":
            from langchain_openai import ChatOpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        else:
            return jsonify({"error": "Invalid LLM choice"}), 400

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=collection.as_retriever(search_kwargs={"k": 10})
        )
        answer = qa_chain.run(query)
        return render_template("chat.html", query=query, answer=answer, current_llm=current_llm)
    else:
        return render_template("chat.html", query="", answer="", current_llm=current_llm)
