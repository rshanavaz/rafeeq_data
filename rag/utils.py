from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STORE_DIR = PROJECT_ROOT / "store"
INDEX_DIR = STORE_DIR / "faiss_index"


def ensure_directories() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Return a HuggingFace embeddings model suitable for CPU.

    Defaults to a solid, lightweight model if none provided.
    """
    model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    # device automatically chosen; default is CPU in most envs
    return HuggingFaceEmbeddings(model_name=model_name)


def save_faiss_index(vectorstore: FAISS, index_dir: Path | str = INDEX_DIR) -> None:
    Path(index_dir).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def load_faiss_index(
    embeddings: HuggingFaceEmbeddings, index_dir: Path | str = INDEX_DIR
) -> FAISS:
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
