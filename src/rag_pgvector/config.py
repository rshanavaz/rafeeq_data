from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str
    pg_collection: str

    hf_token: str
    hf_llm_repo_id: str
    hf_embedding_model: str


def load_settings(env_file: Optional[str] = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    return Settings(
        pg_host=os.getenv("PG_HOST", "localhost"),
        pg_port=int(os.getenv("PG_PORT", "5432")),
        pg_database=os.getenv("PG_DATABASE", "rag_db"),
        pg_user=os.getenv("PG_USER", "postgres"),
        pg_password=os.getenv("PG_PASSWORD", "postgres"),
        pg_collection=os.getenv("PG_COLLECTION", "pdf_rag"),
        hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
        hf_llm_repo_id=os.getenv("HF_LLM_REPO_ID", "HuggingFaceH4/zephyr-7b-beta"),
        hf_embedding_model=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
