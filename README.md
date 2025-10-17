## LangChain RAG Demo

A minimal Retrieval-Augmented Generation (RAG) application using LangChain, FAISS, and HuggingFace embeddings.

### Setup
- Optional but recommended: create a virtualenv
```bash
python -m venv .venv && source .venv/bin/activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

### Ingest data
Put your `.txt`/`.md` files under `data/`. Then build the vector index:
```bash
python -m rag.ingest
```
The FAISS index is stored under `store/faiss_index`.

### Query
Use OpenAI if you have an API key, Ollama if installed, or a placeholder fallback for offline smoke tests.
```bash
# OpenAI (set your key)
export OPENAI_API_KEY=sk-...
python -m rag.query --q "What is this project about?"

# Ollama (if you have it running)
python -m rag.query --q "Summarize the docs" --top_k 6

# Offline fallback
python -m rag.query --q "What does this RAG demo do?"
```

### Notes
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2` (downloads on first use).
- You can change the embedding model in `rag/utils.py`.
- For production, consider moving to a persistent vector DB (e.g., Milvus, PgVector) and robust prompting/evaluation.
