"""Tests for agent nodes with mocked LLM calls."""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.state import MedicalAssistantState
from src.prompts import (
    RouterDecision,
    RelevanceDecision,
    OUT_OF_SCOPE_RESPONSE,
    NOT_FOUND_RESPONSE,
    MEDICAL_DISCLAIMER,
)
from src.agents import (
    out_of_scope_node,
    router_node,
    reasoning_node,
    answer_node,
    clear_session_caches,
)


def get_base_state(query: str = "Test question") -> MedicalAssistantState:
    return {
        "query": query,
        "chat_history": [],
        "route": "document_qa",
        "route_reason": None,
        "rewritten_query": None,
        "documents": [],
        "relevance_ok": False,
        "retry_count": 0,
        "reasoning_notes": None,
        "answer": "",
        "sources": [],
        "llm_calls": 0,
        "node_latencies": {},
    }


def test_out_of_scope_node_static_response():
    state = get_base_state("Write python code")
    res = out_of_scope_node(state)
    assert res["answer"] == OUT_OF_SCOPE_RESPONSE
    assert len(res["sources"]) == 0


@patch("src.agents.call_llm_with_retry")
def test_router_node_llm_out_of_scope(mock_call):
    clear_session_caches()
    mock_call.return_value = RouterDecision(
        route="out_of_scope",
        reason="Query asks for Python code unrelated to medical guidelines."
    )
    state = get_base_state("Write a quicksort function")
    res = router_node(state)
    assert res["route"] == "out_of_scope"
    assert "Python code" in res["route_reason"]


@patch("src.agents.call_llm_with_retry")
def test_router_node_llm_structured_success(mock_call):
    clear_session_caches()
    mock_call.return_value = RouterDecision(
        route="document_qa",
        reason="Clinical guideline inquiry regarding hypertension."
    )
    state = get_base_state("What are first line antihypertensives?")
    res = router_node(state)
    assert res["route"] == "document_qa"
    assert res["route_reason"] == "Clinical guideline inquiry regarding hypertension."


@patch("src.agents.call_llm_with_retry")
def test_router_node_fallback_on_failure(mock_call):
    clear_session_caches()
    # Mock LLM raising unexpected exception
    mock_call.side_effect = RuntimeError("API unavailable")
    state = get_base_state("What are diabetes thresholds?")
    res = router_node(state)
    # Must fallback safely to document_qa
    assert res["route"] == "document_qa"
    assert "Fallback" in res["route_reason"]


@patch("src.agents.call_llm_with_retry")
def test_answer_node_with_documents(mock_call):
    mock_call.return_value = AIMessage(
        content="First-line antihypertensive agents are ACE inhibitors, ARBs, and CCBs [WHO Guidelines, p.23]."
    )
    docs = [
        Document(
            page_content="WHO recommends ACE inhibitors, ARBs, and CCBs as first-line agents.",
            metadata={"title": "WHO Guidelines", "page": 23, "source": "who.pdf"}
        )
    ]
    state = get_base_state()
    state["documents"] = docs
    state["relevance_ok"] = True
    state["reasoning_notes"] = "Key facts extracted."

    res = answer_node(state)
    assert "First-line antihypertensive agents" in res["answer"]
    assert "[WHO Guidelines, p.23]" in res["answer"]
    assert MEDICAL_DISCLAIMER in res["answer"]


def test_answer_node_not_found_fallback():
    state = get_base_state()
    state["documents"] = []
    state["relevance_ok"] = False

    res = answer_node(state)
    assert res["answer"] == NOT_FOUND_RESPONSE
    assert len(res["sources"]) == 0

