from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.llms import FakeListLLM

from .utils import INDEX_DIR, get_embeddings, load_faiss_index


def choose_llm():
    """Pick an available LLM provider.

    Priority:
    1) OpenAI via OPENAI_API_KEY
    2) Ollama if available locally
    3) Fallback FakeListLLM (for offline smoke tests)
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception:
            pass

    if shutil.which("ollama") or os.environ.get("OLLAMA_BASE_URL"):
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(model="llama3.1:8b", temperature=0)
        except Exception:
            pass

    # Deterministic minimal fallback for smoke testing
    return FakeListLLM(responses=[
        "This is a placeholder answer generated without a real LLM. "
        "To get better answers, set OPENAI_API_KEY or run Ollama."
    ])


def format_docs(docs: List) -> str:
    return "\n\n".join(d.page_content for d in docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("-q", "--question", required=True, help="Your question")
    parser.add_argument("-k", "--top_k", type=int, default=4, help="Top K to retrieve")
    args = parser.parse_args()

    embeddings = get_embeddings()
    if not Path(INDEX_DIR).exists():
        raise SystemExit(
            f"Index not found at {INDEX_DIR}. Run `python -m rag.ingest` first."
        )

    vectorstore = load_faiss_index(embeddings, INDEX_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.top_k})

    llm = choose_llm()

    if isinstance(llm, FakeListLLM):
        # Simple offline path without full chain; just show context and a stub answer
        docs = retriever.invoke(args.question)
        context = format_docs(docs)[:1200]
        print(llm.invoke(""))
        print("\n---\nTop sources:\n")
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "unknown")
            print(f"{i}. {source}")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided context to answer the user's question. If the answer isn't in the context, say you don't know.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    answer = chain.invoke(args.question)
    print(answer)

    # Show sources
    docs = retriever.invoke(args.question)
    if docs:
        print("\n---\nTop sources:\n")
        for i, d in enumerate(docs, 1):
            source = d.metadata.get("source", "unknown")
            print(f"{i}. {source}")


if __name__ == "__main__":
    main()
