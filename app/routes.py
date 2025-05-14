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

from scripts.chromadb_handler import ChromaDBHandler
from scripts.pineconedb_handler import PineconeHandler
from scripts.pgvector_hander import PGVectorHandler
from scripts.data_loader import load_medicare_data
from scripts.text_utils import create_dynamic_embedding_text

# Define the USER_CONFIG global variable with default settings.
USER_CONFIG = {
    'dataset': 'Medicare',
    'vector_db': 'ChromaDB',
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'llm': 'deepseek-r1',
    'openai_api_token': '',
    'huggingface_api_token': '',
    
    # ChromaDB specific settings
    'chroma_settings': {
        'collection_name': 'medicare',
        'persist_directory': './chroma_db',
        'collection_metadata': {'description': 'Medicare provider data'},
        'embedding_function': 'sentence-transformers/all-MiniLM-L6-v2',
    },

    # PineconeDB specific settings
    'pinecone_settings': {
        'index_name': '',
        'api_key': '',
    },

    # PGVectorDB specific settings
    'pgvector_settings': {
        'connection_string': 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain',
        'collection_name': 'documents',
        'batch_size': 500,
    }
}

main = Blueprint("main", __name__)

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
        USER_CONFIG['embedding_model'] = request.form.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
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
            
            try:
                # Initialize ChromaDB handler with new settings
                chromadb_handler = ChromaDBHandler(
                    persist_directory=USER_CONFIG['chroma_settings']['persist_directory'],
                    collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']
                )
                # Update the global collection reference for use in other routes
                global collection
                collection = chromadb_handler.vector_store
            except Exception as e:
                print(f"Warning: Could not initialize ChromaDB with new settings: {str(e)}")
        
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
        
        # Update PGVector settings if PGVectorDB is selected
        if USER_CONFIG['vector_db'] == 'PGVectorDB':
            USER_CONFIG['pgvector_settings'].update({
                'connection_string': request.form.get("pgvector_connection", 
                    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"),
                'collection_name': request.form.get("pgvector_collection", "documents"),
                'batch_size': int(request.form.get("pgvector_batch_size", 500))
            })
            
            try:
                # Test connection with new settings
                pgvector_handler = PGVectorHandler(
                    connection_string=USER_CONFIG['pgvector_settings']['connection_string'],
                    collection_name=USER_CONFIG['pgvector_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']
                )
            except Exception as e:
                print(f"Warning: Could not initialize PGVector with new settings: {str(e)}")

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
            # PineconeDB processing
            if USER_CONFIG['vector_db'] == 'PineconeDB':
                try:
                    pinecone_handler = PineconeHandler(
                        api_key=USER_CONFIG['pinecone_settings']['api_key'],
                        index_name=USER_CONFIG['pinecone_settings']['index_name'],
                        embedding_provider=USER_CONFIG['pinecone_settings'].get('embedding_provider', 'pinecone'),
                        model_name=USER_CONFIG['pinecone_settings'].get('model', 'multilingual-e5-large')
                    )
                    doc_size = pinecone_handler.process_documents(data_df)
                    return render_template("configure_embeddings_result.html",
                            success=True,
                            count=doc_size,
                            fields=selected_fields)
                except Exception as e:
                    return render_template("configure_embeddings_result.html",
                                        error=f"Pinecone error: {str(e)}",
                                        fields=[])
            # PGVector processing
            elif USER_CONFIG['vector_db'] == 'PGVectorDB':
                try:
                    pgvector_handler = PGVectorHandler(
                        connection_string=USER_CONFIG['pgvector_settings']['connection_string'],
                        collection_name=USER_CONFIG['pgvector_settings']['collection_name'],
                        embedding_model=USER_CONFIG['embedding_model']
                    )
                    doc_size = pgvector_handler.process_documents(
                        data_df,
                        batch_size=USER_CONFIG['pgvector_settings']['batch_size']
                    )
                    return render_template("configure_embeddings_result.html",
                                           success=True,
                                           count=doc_size,
                                           fields=selected_fields)
                except Exception as e:
                    return render_template("configure_embeddings_result.html",
                                        error=f"PGVector error: {str(e)}",
                                        fields=[])
            else:
                # ChromaDB processing
                try:
                    chromadb_handler = ChromaDBHandler(
                        persist_directory=USER_CONFIG['chroma_settings']['persist_directory'],
                        collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                        embedding_model=USER_CONFIG['embedding_model']
                    )
                    doc_size = chromadb_handler.process_documents(
                        data_df,
                        text_column='text'
                    )
                    return render_template("configure_embeddings_result.html",
                                       success=True,
                                       count=doc_size,
                                       fields=selected_fields)
                except Exception as e:
                    return render_template("configure_embeddings_result.html",
                                        error=f"ChromaDB error: {str(e)}",
                                        fields=[])
                                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return render_template("configure_embeddings_result.html",
                                error=f"Error processing documents: {str(e)}\n{error_traceback}",
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
                            available_fields=available_fields,
                            config=USER_CONFIG)
        
    except Exception as e:
        return render_template("configure_embeddings_result.html",
                            error=f"Error loading dataset: {str(e)}",
                            fields=[])


# Route: Query Documents
@main.route("/query", methods=["GET"])
def query_docs():
    query_text = request.args.get("q", "")
    if query_text:
        try:
            # Initialize appropriate handler based on vector_db setting
            if USER_CONFIG['vector_db'] == 'PGVectorDB':
                handler = PGVectorHandler(
                    connection_string=USER_CONFIG['pgvector_settings']['connection_string'],
                    collection_name=USER_CONFIG['pgvector_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']
                )
            elif USER_CONFIG['vector_db'] == 'PineconeDB':
                handler = PineconeHandler(
                    api_key=USER_CONFIG['pinecone_settings']['api_key'],
                    index_name=USER_CONFIG['pinecone_settings']['index_name'],
                    embedding_provider=USER_CONFIG['pinecone_settings'].get('embedding_provider', 'pinecone'),
                    model_name=USER_CONFIG['pinecone_settings'].get('model', 'multilingual-e5-large')
                )
            else:
                # Default to ChromaDB
                handler = ChromaDBHandler(
                    persist_directory=USER_CONFIG['chroma_settings']['persist_directory'],
                    collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']
                )

            # Perform similarity search using the handler
            results = handler.similarity_search(query_text, k=5)
            return render_template("query.html", query=query_text, results=results)

        except Exception as e:
            error_message = f"Error performing search: {str(e)}"
            return render_template("query.html", query=query_text, error=error_message)
    else:
        # Render the query form if no query parameter is provided
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
            if USER_CONFIG['vector_db'] == 'PGVectorDB':
                pgvector_handler = PGVectorHandler(
                    connection_string=USER_CONFIG['pgvector_settings']['connection_string'],
                    collection_name=USER_CONFIG['pgvector_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']  # Using global embedding model
                )
                retriever = pgvector_handler.get_retriever()

            elif USER_CONFIG['vector_db'] == 'PineconeDB':
                pinecone_handler = PineconeHandler(
                    api_key=USER_CONFIG['pinecone_settings']['api_key'],
                    index_name=USER_CONFIG['pinecone_settings']['index_name'],
                    embedding_provider=USER_CONFIG['pinecone_settings'].get('embedding_provider', 'pinecone'),
                    model_name=USER_CONFIG['pinecone_settings'].get('model', 'multilingual-e5-large')
                )
                retriever = pinecone_handler.get_retriever()

            else:
                # ChromaDB
                chromadb_handler = ChromaDBHandler(
                    persist_directory=USER_CONFIG['chroma_settings']['persist_directory'],
                    collection_name=USER_CONFIG['chroma_settings']['collection_name'],
                    embedding_model=USER_CONFIG['embedding_model']
                )
                retriever = chromadb_handler.get_retriever()

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