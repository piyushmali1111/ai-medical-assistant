"""End-to-end integration tests for the LangGraph workflow with mocked LLM."""

from unittest.mock import patch
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from src.graph import run_query
from src.prompts import RouterDecision, RelevanceDecision, NOT_FOUND_RESPONSE
from src.agents import clear_session_caches


@patch("src.agents.call_llm_with_retry")
def test_graph_out_of_scope_route(mock_call):
    """Verify out-of-scope query routes to out_of_scope node."""
    clear_session_caches()
    mock_call.return_value = RouterDecision(
        route="out_of_scope",
        reason="Query asks for Python code unrelated to medical guidelines."
    )
    res = run_query("Write a python quicksort function")
    assert res["route"] == "out_of_scope"
    assert "specialized exclusively" in res["answer"]
    assert "AI Medical Assistant" in res["answer"]


@patch("src.agents.call_llm_with_retry")
def test_graph_document_qa_route_end_to_end(mock_call):
    """Verify full document_qa path through router -> retriever -> reasoning -> answer."""
    clear_session_caches()

    def mock_llm_side_effect(runnable, prompt_messages):
        prompt_text = str(prompt_messages)
        if "Evidence-Based Answer" in prompt_text or "ANSWER_USER_PROMPT" in prompt_text:
            return AIMessage(content="WHO recommends pharmacological treatment at >= 140/90 mmHg [WHO Guidelines, p.23].")
        elif "Reasoning Notes:" in prompt_text and "REASONING_USER_PROMPT" in prompt_text or "clinical reasoning" in prompt_text:
            return AIMessage(content="Extracted facts: Initiation threshold is >= 140/90 mmHg.")
        elif "relevance" in prompt_text.lower():
            return RelevanceDecision(is_relevant=True, reason="Passages discuss blood pressure cutoffs")
        elif "clinical triage" in prompt_text or "RouterDecision" in str(type(runnable)):
            return RouterDecision(route="document_qa", reason="Hypertension guideline query")
        else:
            return AIMessage(content="WHO recommends pharmacological treatment at >= 140/90 mmHg [WHO Guidelines, p.23].")

    mock_call.side_effect = mock_llm_side_effect

    res = run_query("What is the blood pressure threshold for hypertension treatment?")
    assert res["route"] == "document_qa"
    assert "WHO recommends pharmacological treatment" in res["answer"]
    assert len(res["sources"]) > 0
    assert res["llm_calls"] >= 1


@patch("src.agents.query_vector_store")
@patch("src.agents.call_llm_with_retry")
def test_graph_unanswerable_path(mock_call, mock_query):
    """Verify that empty/irrelevant retrieval routes to answer and gives not found response."""
    clear_session_caches()
    # Mock vector store returning no relevant docs
    mock_query.return_value = []

    def mock_side_effect(runnable, prompt_messages):
        prompt_text = str(prompt_messages)
        if "Router" in str(type(runnable)) or "clinical triage" in prompt_text:
            return RouterDecision(route="document_qa", reason="Medical question")
        elif "relevance" in prompt_text.lower():
            return RelevanceDecision(is_relevant=False, reason="Topic not covered in guidelines")
        else:
            return AIMessage(content="Alternative query rewrite")

    mock_call.side_effect = mock_side_effect

    res = run_query("What is the surgical dose of fentanyl for deep brain stimulation?")
    assert res["route"] == "document_qa"
    assert NOT_FOUND_RESPONSE.split()[0] in res["answer"]
    assert res["sources"] == []
