from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .utils import DATA_DIR, INDEX_DIR, ensure_directories, get_embeddings, save_faiss_index


def load_raw_documents(data_dir: Path) -> list:
    documents: list = []
    # Load .txt and .md files
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(data_dir), glob=pattern, loader_cls=TextLoader, show_progress=True
        )
        documents.extend(loader.load())
    return documents


def build_and_persist_index() -> None:
    ensure_directories()

    raw_docs = load_raw_documents(DATA_DIR)
    if not raw_docs:
        raise SystemExit(
            f"No documents found under {DATA_DIR}. Add .txt or .md files and retry."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=120, add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_faiss_index(vectorstore, INDEX_DIR)
    print(f"Built FAISS index with {len(chunks)} chunks. Saved to: {INDEX_DIR}")


if __name__ == "__main__":
    build_and_persist_index()
