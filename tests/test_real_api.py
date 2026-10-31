"""Live end-to-end API tests against real Gemini service.

These tests run ONLY when GOOGLE_API_KEY is configured in the environment or .env.
"""

import os
import pytest
from src.graph import run_query
from src.agents import clear_session_caches

has_api_key = bool(os.getenv("GOOGLE_API_KEY", "").strip())


@pytest.mark.skipif(not has_api_key, reason="Real-API test requires a valid GOOGLE_API_KEY in .env")
def test_real_gemini_emergency():
    clear_session_caches()
    res = run_query("I have sudden crushing chest pain radiating to my jaw and I am sweating heavily")
    assert res["route"] == "emergency"
    assert "EMERGENCY" in res["answer"].upper()


@pytest.mark.skipif(not has_api_key, reason="Real-API test requires a valid GOOGLE_API_KEY in .env")
def test_real_gemini_out_of_scope():
    clear_session_caches()
    res = run_query("Can you write a python script to sort a list of numbers?")
    assert res["route"] == "out_of_scope"
    assert "Medical Assistant" in res["answer"]


@pytest.mark.skipif(not has_api_key, reason="Real-API test requires a valid GOOGLE_API_KEY in .env")
def test_real_gemini_document_qa():
    clear_session_caches()
    res = run_query("What are the recommended first line drug classes for hypertension according to WHO?")
    assert res["route"] == "document_qa"
    assert len(res["sources"]) > 0
    assert "WHO" in res["answer"] or "hypertension" in res["answer"].lower()
    # Check for citation format
    assert "[" in res["answer"] and "p." in res["answer"]

