# routes.py
# The routes.py file contains the route definitions for the Flask application. It defines the behavior of the application when a user accesses different URLs or endpoints.

import os
import getpass
from flask import Blueprint, request, jsonify, render_template
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
    return render_template("index.html", query="", answer="", contexts="")

# Route: Configure Embeddings
@main.route("/configure_embeddings", methods=["GET", "POST"])
def configure_embeddings():
    if request.method == "POST":
        # Get form data
        selected_fields = request.form.getlist("selected_fields")
        join_option = request.form.get("join_option") == "true"
        separator = request.form.get("separator", " ")
        
        
        # Load Medicare data
        ny_data = load_medicare_data()
        
        if not selected_fields:
            return render_template(
                "configure_embeddings_result.html",
                error="Please select at least one field to embed.",
                fields=[]
            )
        else:
            print(selected_fields)
        
        # Process documents using the global vectorstore
        docs = upsert_medicare_documents(
            collection=collection,
            data_df=ny_data,
            selected_fields=selected_fields,
            join_option=join_option,
            separator=separator
        )
        
        # Render a result template so the user sees a friendly success message.
        return render_template("configure_embeddings_result.html",
                               count=len(docs),
                               fields=selected_fields)
    
    # Handle GET request
    data_df = load_medicare_data()
    available_fields = data_df.columns.tolist()
    return render_template(
        "configure_embeddings.html",
        available_fields=available_fields
    )

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
    if query:
        if MODEL_CHOICE.lower() == "deepseek":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="deepseek-r1", temperature=0.0)
        elif MODEL_CHOICE.lower() == "llama2":
            from langchain_ollama import ChatOllama
            llm = ChatOllama(model="llama2", temperature=0.0)
        elif MODEL_CHOICE.lower() == "openai":
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
            return jsonify({"error": "Invalid MODEL_CHOICE"}), 400

        coll = get_or_create_collection(CHROMADB_PERSIST_DIR, CHROMADB_COLLECTION_NAME, embedding_model)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=coll.as_retriever(search_kwargs={"k": 10})
        )
        answer = qa_chain.run(query)
        return render_template("chat.html", query=query, answer=answer)
    else:
        # Render the chat form if no query parameter is provided.
        return render_template("chat.html", query="", answer="")