"""LangGraph State definition for the AI Medical Assistant workflow."""

from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.documents import Document


class MedicalAssistantState(TypedDict):
    """
    Complete state object tracked throughout the LangGraph workflow.
    """
    # User interaction
    query: str
    chat_history: List[Dict[str, str]] # [{'role': 'user'|'assistant', 'content': '...'}]

    # Routing
    route: str  # 'document_qa' | 'out_of_scope'
    route_reason: Optional[str]

    # Query Rewriting & Retrieval
    rewritten_query: Optional[str]
    documents: List[Document]
    relevance_ok: bool
    retry_count: int

    # Clinical Analysis & Answer Generation
    reasoning_notes: Optional[str]
    answer: str
    sources: List[Dict[str, Any]] # [{'file': '...', 'title': '...', 'page': 1, 'snippet': '...'}]

    # Observability & Metrics
    llm_calls: int
    node_latencies: Dict[str, float]

