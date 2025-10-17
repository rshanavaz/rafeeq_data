from __future__ import annotations

from typing import Sequence

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import load_settings
from .utils.db import PgConnectionInfo, get_connection, ensure_pgvector_enabled


def build_embeddings() -> HuggingFaceEmbeddings:
    settings = load_settings()
    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)


def get_pgvector_connection_string() -> str:
    settings = load_settings()
    return (
        f"postgresql+psycopg://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
    )


def get_pgvector_collection_name() -> str:
    settings = load_settings()
    return settings.pg_collection


def _ensure_vector_extension() -> None:
    settings = load_settings()
    info = PgConnectionInfo(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_database,
        user=settings.pg_user,
        password=settings.pg_password,
    )
    with get_connection(info) as conn:
        ensure_pgvector_enabled(conn)


def upsert_documents(documents: Sequence[Document]) -> PGVector:
    _ensure_vector_extension()
    connection_string = get_pgvector_connection_string()
    collection_name = get_pgvector_collection_name()
    embeddings = build_embeddings()

    vs = PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        use_jsonb=True,
    )
    vs.add_documents(list(documents))
    return vs


def as_retriever() -> PGVector:
    _ensure_vector_extension()
    connection_string = get_pgvector_connection_string()
    collection_name = get_pgvector_collection_name()
    embeddings = build_embeddings()

    return PGVector(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_string=connection_string,
        use_jsonb=True,
    )
