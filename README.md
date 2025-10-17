## RAG with LangChain + Hugging Face + pgvector

This project ingests local PDFs, splits text with LangChain splitters, embeds using a Hugging Face model, and stores vectors in PostgreSQL with the `pgvector` extension. A simple CLI lets you query the knowledge base using a Hugging Face LLM.

### Prerequisites
- Python 3.9+
- PostgreSQL 15+ with `pgvector` extension installed and enabled
- `psql` client recommended for DB checks (optional)
- Hugging Face token (for model access) 

### Setup
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure environment:
```bash
cp .env.example .env
# edit .env with your DB credentials and HF token
```
4. Prepare DB (if needed):
```bash
# Ensure pgvector extension exists in your database
# CREATE EXTENSION IF NOT EXISTS vector;
```

### Ingest PDFs
Place PDFs in `data/pdfs/` then run:
```bash
python -m scripts.ingest --pdf-dir data/pdfs
```

### Ask Questions
Run the RAG CLI:
```bash
python -m scripts.query "What does the document say about X?"
```

### Notes
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` by default
- LLM: `HuggingFaceH4/zephyr-7b-beta` via `HuggingFaceEndpoint`
- You can change models via `.env`
