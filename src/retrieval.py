"""Vector retrieval and document citation formatting utilities."""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import (
    DATA_CHROMA_DIR,
    INDEX_INFO_FILE,
    CHROMA_COLLECTION_NAME,
    TOP_K,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    get_document_title,
)
from src.utils import logger
from src.llm import get_embeddings


def check_index_consistency(persist_dir: Path = DATA_CHROMA_DIR) -> Dict[str, Any]:
    """
    Verify that the vector store exists and the configured embedding model
    matches what the index was built with.
    """
    index_file = persist_dir / "index_info.json"
    if not index_file.exists():
        logger.warning(
            "Vector store index metadata not found at '%s'. "
            "Please run 'python -m src.ingest --reset' to build the index.",
            index_file,
        )
        return {}

    try:
        with open(index_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        indexed_model = info.get("embedding_model")
        indexed_provider = info.get("embedding_provider")

        if (
            indexed_model
            and indexed_model != EMBEDDING_MODEL
            or (indexed_provider and indexed_provider != EMBEDDING_PROVIDER)
        ):
            logger.warning(
                "WARNING: Configured embedding (%s: %s) differs from index "
                "(%s: %s). Results may be inaccurate. "
                "Run 'python -m src.ingest --reset' to rebuild the index.",
                EMBEDDING_PROVIDER,
                EMBEDDING_MODEL,
                indexed_provider,
                indexed_model,
            )
        return info
    except Exception as e:
        logger.warning("Could not read index info: %s", e)
        return {}


def get_vector_store(
    persist_dir: Path = DATA_CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> Chroma:
    """Return a persistent Chroma vector store instance."""
    check_index_consistency(persist_dir)
    embeddings = get_embeddings(
        provider=embedding_provider or EMBEDDING_PROVIDER,
        model_name=embedding_model or EMBEDDING_MODEL,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def get_retriever(
    top_k: int = TOP_K,
    persist_dir: Path = DATA_CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    """Return a vector retriever configured with top_k."""
    vector_store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Return a standardized inline citation string derived from chunk metadata.
    Format: [Title, p.X]
    """
    title = metadata.get("title") or get_document_title(metadata.get("source", "Document"))
    page = metadata.get("page", "?")
    return f"[{title}, p.{page}]"


def format_documents_for_prompt(docs: List[Document]) -> str:
    """
    Format a list of retrieved Document objects into a clean, numbered context
    block with clear inline citation markers for the LLM prompt.
    """
    if not docs:
        return "No relevant medical context retrieved."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        citation = format_citation(doc.metadata)
        source_file = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        block = (
            f"--- Document Context [{idx}] ---\n"
            f"Source: {source_file} (Page {page})\n"
            f"Citation Tag: {citation}\n"
            f"Text:\n{doc.page_content.strip()}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


def extract_sources_summary(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract a unique, structured list of source metadata dictionaries from docs.
    """
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata
        file = meta.get("source", "unknown")
        page = meta.get("page", 1)
        title = meta.get("title", get_document_title(file))
        key = (file, page)
        if key not in seen:
            seen.add(key)
            sources.append(
                {
                    "file": file,
                    "title": title,
                    "page": page,
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
            )
    return sources


def query_vector_store(query: str, k: int = TOP_K) -> List[Document]:
    """Retrieve top-k documents for a query string."""
    retriever = get_retriever(top_k=k)
    return retriever.invoke(query)


def main():
    parser = argparse.ArgumentParser(description="Query the medical vector store directly.")
    parser.add_argument("query", type=str, help="Question or medical search query")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print(f"\nQuery: '{args.query}' (top_k={args.k})\n")
    docs = query_vector_store(args.query, k=args.k)

    if not docs:
        print("No documents retrieved.")
        return

    print(f"Retrieved {len(docs)} chunks:\n")
    for i, doc in enumerate(docs, 1):
        citation = format_citation(doc.metadata)
        print(f"[{i}] {citation}")
        print(f"    File: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
        print(f"    Content preview: {doc.page_content[:250].strip()}...")
        print("-" * 60)


if __name__ == "__main__":
    main()

