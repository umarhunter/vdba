# Vector Database Analysis (VDBA)

VDBA is a Flask-based web application for analyzing and comparing different vector databases using various language models and embedding techniques. This tool allows users to explore document similarity search and question answering capabilities across multiple vector database backends.

## Features

- **Multiple Vector Database Support**
  - ChromaDB
  - PGVector (PostgreSQL)
  - Pinecone

- **Flexible LLM Integration**
  - Local models via Ollama
  - OpenAI API support
  - Configurable model settings

- **Document Analysis**
  - Support for multiple datasets
  - Configurable embedding models
  - Real-time chat interface
  - Thought process visualization

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension (for PGVector support)
- Docker (optional, for containerized databases)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vdba.git
cd vdba
```

2. Create and activate a virtual environment (or use anaconda/mamba):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your environment variables in `.env`:
```
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

### Database Setup

#### PostgreSQL with pgvector
```bash
docker pull pgvector/pgvector
docker run -d --name pgvector -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=langchain -p 6024:5432 pgvector/pgvector     
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Navigate to `http://localhost:5000` in your web browser

3. Configure your settings:
   - Select vector database
   - Choose embedding model
   - Select LLM
   - Upload or select dataset

4. Start analyzing documents through the chat interface

## Project Structure

```
vdba/
├── app/
│   ├── templates/    # HTML templates
│   ├── static/       # Static assets
│   └── routes.py     # Flask routes
├── scripts/
│   ├── chromadb_handler.py
│   ├── pgvector_hander.py
│   ├── pineconedb_handler.py
│   └── data_loader.py
└── requirements.txt
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/featureXYZ`)
3. Commit your changes (`git commit -m 'Add some featureXYZ'`)
4. Push to the branch (`git push origin feature/featureXYZ`)
5. Open a Pull Request

## License

This project is licensed under the MIT [License](LICENSE)

## Acknowledgments

- LangChain for vector store integrations
- Sentence Transformers for embedding models
- ChromaDB, PGVector, and Pinecone for vector database support