"""Tests for text cleaning, document loading, and chunking in ingestion."""

from pathlib import Path
from langchain_core.documents import Document

from src.utils import clean_page_text
from src.ingest import split_documents
from src.config import get_document_title


def test_clean_page_text_normalizes_whitespace():
    raw = "This is a   test   line.\n\n\n\nAnother paragraph with   spaces."
    cleaned = clean_page_text(raw)
    assert "This is a test line." in cleaned
    assert "Another paragraph with spaces." in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_page_text_rejoins_hyphenated_words():
    raw = "The pharma-\ncological treatment of hyper-\ntension."
    cleaned = clean_page_text(raw)
    assert "pharmacological" in cleaned
    assert "hypertension" in cleaned


def test_clean_page_text_handles_ligatures():
    # \ufb01 is 'fi', \ufb02 is 'fl'
    raw = "Classi\ufb01cation of blood \ufb02ow."
    cleaned = clean_page_text(raw)
    assert "Classification" in cleaned
    assert "flow" in cleaned


def test_get_document_title_known_and_fallback():
    assert get_document_title("who_hypertension_2021.pdf") == "WHO Guidelines on Pharmacological Treatment of Hypertension in Adults"
    # Fallback derivation
    title = get_document_title("custom_clinical_protocol_2025.pdf")
    assert "Custom Clinical Protocol 2025" in title


def test_split_documents_preserves_metadata_and_adds_chunk_id():
    doc = Document(
        page_content="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 30,
        metadata={"source": "test_doc.pdf", "title": "Test Title", "page": 4},
    )
    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "test_doc.pdf"
        assert chunk.metadata["title"] == "Test Title"
        assert chunk.metadata["page"] == 4
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith("test_doc.pdf_p4_c")

