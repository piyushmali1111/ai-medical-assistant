"""Ingestion pipeline for PDF documents into Chroma vector store."""

import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import (
    DATA_RAW_DIR,
    DATA_CHROMA_DIR,
    INDEX_INFO_FILE,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    get_document_title,
)
from src.utils import clean_page_text, logger
from src.llm import get_embeddings


def load_pdf_pages(file_path: Path) -> List[Document]:
    """
    Load a PDF document page by page, apply light text cleaning,
    drop empty pages, and attach source metadata.
    """
    documents: List[Document] = []
    filename = file_path.name
    doc_title = get_document_title(filename)

    try:
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_page_text(raw_text)
            # Skip empty or near-empty pages (e.g. blank cover backs)
            if not cleaned or len(cleaned.strip()) < 15:
                continue

            doc = Document(
                page_content=cleaned,
                metadata={
                    "source": filename,
                    "title": doc_title,
                    "page": page_idx + 1, # 1-based page numbering
                },
            )
            documents.append(doc)
    except Exception as e:
        logger.error("Failed to load PDF '%s': %s", filename, e)

    return documents


def load_all_raw_documents(raw_dir: Path = DATA_RAW_DIR) -> Tuple[List[Document], int]:
    """
    Scan raw directory for all PDFs and extract clean page-level documents.
    Returns (list of page documents, number of PDF files).
    """
    pdf_files = sorted(list(raw_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'", raw_dir)
        return [], 0

    all_page_docs: List[Document] = []
    for pdf_file in pdf_files:
        page_docs = load_pdf_pages(pdf_file)
        logger.info("Loaded '%s': %d non-empty pages", pdf_file.name, len(page_docs))
        all_page_docs.extend(page_docs)

    return all_page_docs, len(pdf_files)


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page documents into chunks using RecursiveCharacterTextSplitter.
    Preserves and enriches metadata with chunk index.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    # Enrich chunks with deterministic chunk_id
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_p{chunk.metadata['page']}_c{idx}"

    return chunks


def ingest_documents(
    reset: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_provider: str = EMBEDDING_PROVIDER,
    embedding_model: str = EMBEDDING_MODEL,
    persist_dir: Path = DATA_CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
    1. Optionally reset existing persistent Chroma vector store
    2. Load PDFs from data/raw
    3. Clean and chunk documents
    4. Compute embeddings and persist in Chroma
    5. Save index metadata to index_info.json
    """
    start_time = time.time()
    logger.info("Starting ingestion (reset=%s)...", reset)

    persist_dir.mkdir(parents=True, exist_ok=True)
    index_file = persist_dir / "index_info.json"

    if reset and persist_dir.exists():
        logger.info("Reset requested. Clearing existing vector store at '%s'...", persist_dir)
        # Remove old directory contents cleanly
        for item in persist_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except Exception:
                    pass

    # 1. Load documents
    page_docs, num_docs = load_all_raw_documents()
    if not page_docs:
        raise RuntimeError(f"No valid PDF pages found to ingest in '{DATA_RAW_DIR}'.")

    # 2. Chunk documents
    chunks = split_documents(page_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Generated %d chunks from %d pages across %d documents.", len(chunks), len(page_docs), num_docs)

    # 3. Instantiate embeddings
    embeddings = get_embeddings(provider=embedding_provider, model_name=embedding_model)

    # 4. Ingest into Chroma in batches to avoid memory/IPC bottlenecks
    batch_size = 200
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c.metadata["chunk_id"] for c in batch]
        vector_store.add_documents(documents=batch, ids=ids)
        logger.info("Ingested batch %d/%d (%d chunks)...", i + len(batch), len(chunks), len(batch))

    total_time = time.time() - start_time

    # 5. Persist index metadata
    metadata = {
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_documents": num_docs,
        "num_pages": len(page_docs),
        "num_chunks": len(chunks),
        "ingestion_time_seconds": round(total_time, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Ingestion completed: %d documents, %d pages, %d chunks in %.2f seconds.",
        num_docs,
        len(page_docs),
        len(chunks),
        total_time,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma vector store.")
    parser.add_argument("--reset", action="store_true", help="Clear existing Chroma vector store before ingesting.")
    args = parser.parse_args()

    stats = ingest_documents(reset=args.reset)
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Documents ingested: {stats['num_documents']}")
    print(f"Total pages:        {stats['num_pages']}")
    print(f"Total chunks:       {stats['num_chunks']}")
    print(f"Embedding model:    {stats['embedding_model']} ({stats['embedding_provider']})")
    print(f"Chunk size/overlap: {stats['chunk_size']} / {stats['chunk_overlap']}")
    print(f"Total time:         {stats['ingestion_time_seconds']} seconds")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()

