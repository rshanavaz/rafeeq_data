from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from tqdm import tqdm

from rag_pgvector.config import load_settings
from rag_pgvector.utils.loader import load_pdfs_from_dir
from rag_pgvector.utils.text_splitter import split_documents
from rag_pgvector.vectorstore import upsert_documents

app = typer.Typer(add_completion=False)


@app.command()
def main(
    pdf_dir: str = typer.Option("data/pdfs", help="Directory containing PDFs"),
    chunk_size: int = typer.Option(1000, help="Chunk size for splitting"),
    chunk_overlap: int = typer.Option(150, help="Chunk overlap for splitting"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
):
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    settings = load_settings()

    print(
        f"Ingesting PDFs from '{pdf_dir}' into collection '{settings.pg_collection}'"
    )

    documents = load_pdfs_from_dir(pdf_dir)
    print(f"Loaded {len(documents)} pages from PDFs")

    splits = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(splits)} chunks")

    for d in splits:
        d.metadata.setdefault("source", d.metadata.get("source", "pdf"))
        d.metadata.setdefault("collection", settings.pg_collection)

    print("Upserting documents into pgvector... this may take a while")
    upsert_documents(tqdm(splits))
    print("Done.")


if __name__ == "__main__":
    app()
