"""Tests for citation formatting, document prompt preparation, and vector retrieval."""

from langchain_core.documents import Document

from src.retrieval import (
    format_citation,
    format_documents_for_prompt,
    extract_sources_summary,
    query_vector_store,
)


def test_format_citation_standard():
    meta = {"title": "WHO Hypertension Guidelines", "page": 12, "source": "who.pdf"}
    citation = format_citation(meta)
    assert citation == "[WHO Hypertension Guidelines, p.12]"


def test_format_documents_for_prompt():
    docs = [
        Document(
            page_content="First line drugs include ACE inhibitors and CCBs.",
            metadata={"title": "WHO Guidelines", "page": 23, "source": "who_ht.pdf"},
        ),
        Document(
            page_content="Target blood pressure is below 140/90 mmHg.",
            metadata={"title": "WHO Guidelines", "page": 25, "source": "who_ht.pdf"},
        ),
    ]
    prompt_context = format_documents_for_prompt(docs)
    assert "--- Document Context [1] ---" in prompt_context
    assert "Citation Tag: [WHO Guidelines, p.23]" in prompt_context
    assert "First line drugs include ACE inhibitors and CCBs." in prompt_context
    assert "--- Document Context [2] ---" in prompt_context


def test_extract_sources_summary_deduplicates():
    docs = [
        Document(page_content="Text A", metadata={"source": "file1.pdf", "title": "Doc 1", "page": 5}),
        Document(page_content="Text B", metadata={"source": "file1.pdf", "title": "Doc 1", "page": 5}),
        Document(page_content="Text C", metadata={"source": "file2.pdf", "title": "Doc 2", "page": 10}),
    ]
    summary = extract_sources_summary(docs)
    assert len(summary) == 2
    assert summary[0]["file"] == "file1.pdf"
    assert summary[0]["page"] == 5
    assert summary[1]["file"] == "file2.pdf"


def test_query_vector_store_returns_k_documents():
    """Verify retriever returns k results from the Chroma index."""
    k = 3
    results = query_vector_store("hypertension treatment", k=k)
    assert len(results) == k
    for doc in results:
        assert isinstance(doc, Document)
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert len(doc.page_content) > 0

