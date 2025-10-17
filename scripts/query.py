from __future__ import annotations

from typing import Optional

import typer
from dotenv import load_dotenv

from rag_pgvector.config import load_settings
from rag_pgvector.llm import build_hf_llm
from rag_pgvector.vectorstore import as_retriever

app = typer.Typer(add_completion=False)


@app.command()
def main(
    question: str = typer.Argument(..., help="User question"),
    k: int = typer.Option(4, help="Top-k retrieved chunks"),
    env_file: Optional[str] = typer.Option(None, help="Path to .env file"),
):
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    _ = load_settings()  # validate env
    retriever = as_retriever()
    docs = retriever.similarity_search(question, k=k)

    llm = build_hf_llm()

    context = "\n\n".join(f"[source: {d.metadata.get('source')}]\n{d.page_content}" for d in docs)
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    answer = llm.invoke(prompt)
    print(answer)


if __name__ == "__main__":
    app()
