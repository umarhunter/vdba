# routes.py
# The routes.py file contains the route definitions for the Flask application. It defines the behavior of the application when a user accesses different URLs or endpoints.

import os
import chromadb
import getpass
import ollama
import pandas as pd

from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename

from scripts.chromadb_handler import (
    initialize_chroma_client,
    get_or_create_collection,
    upsert_medicare_documents,
    create_dynamic_embedding_text,
    concatenate_fields
)
from scripts.pineconedb_handler import PineconeHandler
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


def get_available_llm_models():
    try:
        response = ollama.list()
        # Extract model names and remove ':latest' suffix
        models = [model.model.replace(':latest', '') for model in response.models]
        return models
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return ["deepseek", "llama2"]  # Fallback default models
    

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
                'api_key': request.form.get("pinecone_api_key", ""),
                'embedding_provider': request.form.get("pinecone_embedding_provider", "pinecone"),
                'model': (request.form.get("pinecone_model") 
                        if request.form.get("pinecone_embedding_provider") == "pinecone"
                        else request.form.get("openai_model")),
                'namespace': 'default'
            })
        
        return redirect(url_for("main.dashboard"))
    else:
        available_models = get_available_llm_models()
        return render_template("settings.html", 
                            config=USER_CONFIG,
                            available_models=available_models)

@main.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    if 'csv-file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['csv-file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'Only CSV files are supported'})
    
    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # Add to custom datasets
        dataset_info = {
            'name': os.path.splitext(filename)[0],
            'path': filepath
        }
        
        # Update USER_CONFIG
        if 'custom_datasets' not in USER_CONFIG:
            USER_CONFIG['custom_datasets'] = []
            
        # Check if dataset already exists
        existing = next((d for d in USER_CONFIG['custom_datasets'] 
                        if d['name'] == dataset_info['name']), None)
        if existing:
            existing.update(dataset_info)
        else:
            USER_CONFIG['custom_datasets'].append(dataset_info)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
# Route: Configure Embeddings
@main.route("/configure_embeddings", methods=["GET", "POST"])
def configure_embeddings():
    if request.method == "POST":
        # Get form data
        selected_fields = request.form.getlist("selected_fields")
        join_option = request.form.get("join_option") == "true"
        separator = request.form.get("separator", " ")
        
        # Load dataset (works with both custom and default datasets)
        try:
            if USER_CONFIG['dataset'] == "Medicare":
                data_df = load_medicare_data()
            else:
                # Load custom dataset
                dataset_info = next((d for d in USER_CONFIG.get('custom_datasets', []) 
                                  if d['name'] == USER_CONFIG['dataset']), None)
                if dataset_info:
                    data_df = pd.read_csv(dataset_info['path'], dtype=str)
                else:
                    raise ValueError(f"Dataset {USER_CONFIG['dataset']} not found")
            
            # Create options dictionary
            options = {
                'join_option': request.form.get("join_option") == "true",
                'separator': request.form.get("separator", " ")
            }

            # Create text column from selected fields
            data_df['text'] = create_dynamic_embedding_text(
                data_df, 
                selected_fields,
                options
            )
            
            # Process documents based on selected vector store
            if USER_CONFIG['vector_db'] == 'PineconeDB':
                try:
                    pinecone_handler = PineconeHandler(
                        api_key=USER_CONFIG['pinecone_settings']['api_key'],
                        index_name=USER_CONFIG['pinecone_settings']['index_name'],
                        embedding_provider=USER_CONFIG['pinecone_settings'].get('embedding_provider', 'pinecone'),
                        model_name=USER_CONFIG['pinecone_settings'].get('model', 'multilingual-e5-large')
                    )
                    docs = pinecone_handler.process_documents(data_df)
                except Exception as e:
                    return render_template("configure_embeddings_result.html",
                                        error=f"Pinecone error: {str(e)}",
                                        fields=[])
            else:
                # ChromaDB processing
                collection = get_or_create_collection(
                    persist=USER_CONFIG['chroma_settings']['persist_directory'],
                    collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                    embedding_model=embedding_model
                )
                docs = collection.add_documents(data_df['text'].tolist())
            
            return render_template("configure_embeddings_result.html",
                                success=True,
                                count=len(docs),
                                fields=selected_fields)
                                
        except Exception as e:
            return render_template("configure_embeddings_result.html",
                                error=f"Error processing documents: {str(e)}",
                                fields=[])
    
    # Handle GET request
    try:
        # Get available fields from current dataset
        if USER_CONFIG['dataset'] == "Medicare":
            data_df = load_medicare_data()
        else:
            dataset_info = next((d for d in USER_CONFIG.get('custom_datasets', []) 
                              if d['name'] == USER_CONFIG['dataset']), None)
            if dataset_info:
                data_df = pd.read_csv(dataset_info['path'], dtype=str)
            else:
                return render_template("configure_embeddings_result.html",
                                    error="Dataset not found",
                                    fields=[])
        
        available_fields = data_df.columns.tolist()
        return render_template("configure_embeddings.html", 
                            available_fields=available_fields)
        
    except Exception as e:
        return render_template("configure_embeddings_result.html",
                            error=f"Error loading dataset: {str(e)}",
                            fields=[])


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
    
    if 'chat_history' not in session:
        session['chat_history'] = []

    if query:
        try:
            # Initialize appropriate retriever based on vector_db setting
            if USER_CONFIG['vector_db'] == 'PineconeDB':
                pinecone_handler = PineconeHandler(
                    api_key=USER_CONFIG['pinecone_settings']['api_key'],
                    index_name=USER_CONFIG['pinecone_settings']['index_name'],
                    embedding_provider=USER_CONFIG['pinecone_settings'].get('embedding_provider', 'pinecone'),
                    model_name=USER_CONFIG['pinecone_settings'].get('model', 'multilingual-e5-large')
                )
                retriever = pinecone_handler.get_retriever()
            else:
                retriever = collection.as_retriever(search_kwargs={"k": 10})

            # Initialize LLM
            selected_llm = USER_CONFIG['llm'].lower()
            try:
                if selected_llm == "openai":
                    from langchain_openai import ChatOpenAI
                    if not os.environ.get("OPENAI_API_KEY"):
                        return jsonify({"error": "OpenAI API key not set"}), 400
                    llm = ChatOpenAI(
                        model="gpt-4",
                        temperature=0,
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
                else:
                    from langchain_ollama import ChatOllama
                    llm = ChatOllama(model=selected_llm, temperature=0.0)
            except Exception as e:
                return jsonify({"error": f"Failed to load model '{selected_llm}': {str(e)}"}), 400

            # Create QA chain with appropriate retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            
            answer = qa_chain.run(query)
            
            # Add to chat history
            session['chat_history'].append({'role': 'user', 'content': query})
            session['chat_history'].append({'role': 'assistant', 'content': answer})
            
            return jsonify({'answer': answer})
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    return render_template("chat.html", 
                         current_llm=current_llm,
                         chat_history=session.get('chat_history', []))