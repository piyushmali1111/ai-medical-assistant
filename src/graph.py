"""LangGraph workflow definition, compilation, and execution runner."""

from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END

from src.state import MedicalAssistantState
from src.agents import (
    router_node,
    retriever_node,
    reasoning_node,
    answer_node,
    emergency_node,
    out_of_scope_node,
)
from src.utils import logger


def route_decision_edge(state: MedicalAssistantState) -> str:
    """Conditional branching edge following the router classification."""
    route = state.get("route", "document_qa")
    if route == "emergency":
        return "emergency"
    if route == "out_of_scope":
        return "out_of_scope"
    return "retriever"


def relevance_decision_edge(state: MedicalAssistantState) -> str:
    """Conditional branching edge following document retrieval and relevance check."""
    if state.get("relevance_ok", False):
        return "reasoning"
    return "answer"


def build_graph() -> StateGraph:
    """Construct and configure the LangGraph workflow structure."""
    workflow = StateGraph(MedicalAssistantState)

    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("emergency", emergency_node)
    workflow.add_node("out_of_scope", out_of_scope_node)

    # Edge: START -> router
    workflow.add_edge(START, "router")

    # Conditional: router -> {retriever, emergency, out_of_scope}
    workflow.add_conditional_edges(
        "router",
        route_decision_edge,
        {
            "retriever": "retriever",
            "emergency": "emergency",
            "out_of_scope": "out_of_scope",
        },
    )

    # Conditional: retriever -> {reasoning, answer (not-found)}
    workflow.add_conditional_edges(
        "retriever",
        relevance_decision_edge,
        {
            "reasoning": "reasoning",
            "answer": "answer",
        },
    )

    # Edge: reasoning -> answer -> END
    workflow.add_edge("reasoning", "answer")
    workflow.add_edge("answer", END)

    # Terminal static paths -> END
    workflow.add_edge("emergency", END)
    workflow.add_edge("out_of_scope", END)

    return workflow


_compiled_graph = None

def get_compiled_graph():
    """Singleton getter for the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_graph()
        _compiled_graph = workflow.compile()
    return _compiled_graph


def run_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Execute an end-to-end query through the LangGraph medical assistant pipeline.
    
    Returns:
        dict containing answer, route, sources, documents, llm_calls, node_latencies
    """
    graph = get_compiled_graph()
    initial_state: MedicalAssistantState = {
        "query": query,
        "chat_history": chat_history or [],
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

    config = {
        "run_name": f"MedicalAssistant: {query[:35]}",
        "tags": ["medical-assistant", "langgraph", "gemini-flash"],
        "metadata": {
            "query_length": len(query),
            "history_turns": len(chat_history or []),
        },
    }

    final_state = graph.invoke(initial_state, config=config)

    return {
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", "document_qa"),
        "route_reason": final_state.get("route_reason"),
        "sources": final_state.get("sources", []),
        "documents": final_state.get("documents", []),
        "llm_calls": final_state.get("llm_calls", 0),
        "node_latencies": final_state.get("node_latencies", {}),
        "rewritten_query": final_state.get("rewritten_query"),
        "reasoning_notes": final_state.get("reasoning_notes"),
    }

