from __future__ import annotations

from langchain_huggingface import HuggingFaceEndpoint

from .config import load_settings


def build_hf_llm() -> HuggingFaceEndpoint:
    settings = load_settings()
    if not settings.hf_token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is required. Set it in your .env."
        )
    # The token is picked up via env var by the client
    return HuggingFaceEndpoint(repo_id=settings.hf_llm_repo_id, temperature=0.2)
