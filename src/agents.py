"""Node agent functions for the LangGraph medical assistant pipeline."""

import time
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

from src.state import MedicalAssistantState
from src.config import (
    ENABLE_RELEVANCE_GRADER,
    ENABLE_SEPARATE_REASONING,
    TOP_K,
)
from src.prompts import (
    RouterDecision,
    RelevanceDecision,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
    RELEVANCE_GRADER_SYSTEM_PROMPT,
    RELEVANCE_GRADER_USER_PROMPT,
    REASONING_SYSTEM_PROMPT,
    REASONING_USER_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    NOT_FOUND_RESPONSE,
    MEDICAL_DISCLAIMER,
)
from src.llm import get_llm, call_llm_with_retry
from src.retrieval import (
    query_vector_store,
    format_documents_for_prompt,
    extract_sources_summary,
)
from src.utils import logger

# In-memory session caches to eliminate redundant LLM calls
_router_cache: Dict[str, RouterDecision] = {}
_rewrite_cache: Dict[str, str] = {}


def clear_session_caches():
    """Clear in-memory router and query rewriter caches."""
    _router_cache.clear()
    _rewrite_cache.clear()


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format recent chat history into a human-readable dialogue string."""
    if not messages:
        return "None"
    formatted = []
    # Use last 4 turns to keep context tight and prevent token waste
    for msg in messages[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def router_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """
    Router node: Classifies query into 'document_qa' or 'out_of_scope' purely using the LLM.
    Uses structured output with retry and safe fallback.
    """
    start_time = time.time()
    query = state["query"].strip()
    history_str = format_chat_history(state.get("chat_history", []))
    cache_key = f"{query.lower()}|{history_str[-100:]}"

    if cache_key in _router_cache:
        decision = _router_cache[cache_key]
        logger.info("Router cache hit: %s (%s)", decision.route, decision.reason)
        return {
            "route": decision.route,
            "route_reason": f"[Cached] {decision.reason}",
            "node_latencies": {**state.get("node_latencies", {}), "router": time.time() - start_time},
        }

    # Route classification purely via LLM structured output
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RouterDecision)
    prompt_messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=ROUTER_USER_PROMPT.format(chat_history=history_str, query=query)),
    ]

    llm_calls = state.get("llm_calls", 0)
    decision = None

    # Attempt with 1 retry on structured parsing failure
    for attempt in range(2):
        try:
            llm_calls += 1
            decision = call_llm_with_retry(structured_llm, prompt_messages)
            if isinstance(decision, RouterDecision):
                break
        except Exception as e:
            logger.warning("Router classification attempt %d failed: %s", attempt + 1, e)

    if not isinstance(decision, RouterDecision):
        logger.warning("Router structured parsing failed. Falling back to default 'document_qa'.")
        decision = RouterDecision(route="document_qa", reason="Fallback default due to parsing error.")

    _router_cache[cache_key] = decision
    latency = time.time() - start_time
    logger.info("Router decided: '%s' in %.2fs (Reason: %s)", decision.route, latency, decision.reason)

    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "llm_calls": llm_calls,
        "node_latencies": {**state.get("node_latencies", {}), "router": latency},
    }


def retriever_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """
    Retriever node:
    1. Rewrites query if conversational context exists
    2. Retrieves top-k chunks from Chroma
    3. Evaluates relevance (if ENABLE_RELEVANCE_GRADER)
    4. Reformulates and retries once if not relevant
    """
    start_time = time.time()
    query = state["query"].strip()
    history = state.get("chat_history", [])
    retry_count = state.get("retry_count", 0)
    llm_calls = state.get("llm_calls", 0)
    history_str = format_chat_history(history)

    # 1. Query Rewriting
    search_query = query
    cache_key = f"{query.lower()}|{history_str[-100:]}|retry_{retry_count}"

    if cache_key in _rewrite_cache:
        search_query = _rewrite_cache[cache_key]
    elif history or retry_count > 0:
        llm = get_llm(temperature=0.0)
        prompt = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=QUERY_REWRITE_USER_PROMPT.format(
                chat_history=history_str,
                query=query,
                is_retry=(retry_count > 0),
            )),
        ]
        try:
            llm_calls += 1
            resp = call_llm_with_retry(llm, prompt)
            rewritten = resp.content.strip()
            if rewritten and len(rewritten) > 3:
                search_query = rewritten
                _rewrite_cache[cache_key] = search_query
        except Exception as e:
            logger.warning("Query rewrite failed, using original query: %s", e)

    # 2. Vector Retrieval
    docs = query_vector_store(search_query, k=TOP_K)
    logger.info("Retrieved %d chunks for search query: '%s'", len(docs), search_query)

    # 3. Relevance Grading
    relevance_ok = len(docs) > 0

    if ENABLE_RELEVANCE_GRADER and docs:
        llm = get_llm(temperature=0.0)
        grader_llm = llm.with_structured_output(RelevanceDecision)
        context_str = "\n\n".join([d.page_content for d in docs[:3]])
        prompt = [
            SystemMessage(content=RELEVANCE_GRADER_SYSTEM_PROMPT),
            HumanMessage(content=RELEVANCE_GRADER_USER_PROMPT.format(
                query=query,
                context=context_str,
            )),
        ]
        try:
            llm_calls += 1
            grade = call_llm_with_retry(grader_llm, prompt)
            if isinstance(grade, RelevanceDecision):
                relevance_ok = grade.is_relevant
                logger.info("Relevance grader verdict: %s (Reason: %s)", relevance_ok, grade.reason)
        except Exception as e:
            logger.warning("Relevance grading failed; falling back to True: %s", e)
            relevance_ok = True

    # 4. If not relevant and retry_count < 1, reformulate and retry once
    if not relevance_ok and retry_count < 1:
        logger.info("Retrieved chunks not relevant. Attempting query reformulation retry...")
        retry_count += 1
        # Reformulate query with different keywords
        llm = get_llm(temperature=0.2)
        retry_prompt = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=QUERY_REWRITE_USER_PROMPT.format(
                chat_history=history_str,
                query=f"{query} (focus on clinical definitions and evidence-based guideline terminology)",
                is_retry=True,
            )),
        ]
        try:
            llm_calls += 1
            resp = call_llm_with_retry(llm, retry_prompt)
            alt_query = resp.content.strip()
            if alt_query:
                search_query = alt_query
                docs = query_vector_store(search_query, k=TOP_K)
                # Second check
                relevance_ok = len(docs) > 0
        except Exception as e:
            logger.warning("Retry search failed: %s", e)

    sources = extract_sources_summary(docs) if relevance_ok else []
    latency = time.time() - start_time

    return {
        "rewritten_query": search_query,
        "documents": docs if relevance_ok else [],
        "relevance_ok": relevance_ok,
        "retry_count": retry_count,
        "sources": sources,
        "llm_calls": llm_calls,
        "node_latencies": {**state.get("node_latencies", {}), "retriever": latency},
    }


def reasoning_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """
    Reasoning node: Extracts factual answers, identifies conflicts or gaps.
    Can be bypassed if ENABLE_SEPARATE_REASONING is False.
    """
    start_time = time.time()
    docs = state.get("documents", [])
    query = state["query"]
    llm_calls = state.get("llm_calls", 0)

    if not ENABLE_SEPARATE_REASONING or not docs:
        return {
            "reasoning_notes": None,
            "node_latencies": {**state.get("node_latencies", {}), "reasoning": time.time() - start_time},
        }

    context_str = format_documents_for_prompt(docs)
    llm = get_llm(temperature=0.0)
    prompt = [
        SystemMessage(content=REASONING_SYSTEM_PROMPT),
        HumanMessage(content=REASONING_USER_PROMPT.format(query=query, context=context_str)),
    ]

    notes = ""
    try:
        llm_calls += 1
        resp = call_llm_with_retry(llm, prompt)
        notes = resp.content.strip()
    except Exception as e:
        logger.warning("Reasoning node failed; continuing to answer node: %s", e)
        notes = "Extracted relevant facts from retrieved clinical documents."

    latency = time.time() - start_time
    return {
        "reasoning_notes": notes,
        "llm_calls": llm_calls,
        "node_latencies": {**state.get("node_latencies", {}), "reasoning": latency},
    }


def answer_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """
    Answer node: Generates grounded, cited final answer using only context and notes.
    If context is insufficient, falls back to NOT_FOUND_RESPONSE.
    """
    start_time = time.time()
    docs = state.get("documents", [])
    relevance_ok = state.get("relevance_ok", False)
    query = state["query"]
    notes = state.get("reasoning_notes") or "Direct extraction from provided clinical guidelines."
    llm_calls = state.get("llm_calls", 0)

    # If documents were not found or not relevant, return safe grounded fallback
    if not relevance_ok or not docs:
        return {
            "answer": NOT_FOUND_RESPONSE,
            "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "answer": time.time() - start_time},
        }

    context_str = format_documents_for_prompt(docs)
    llm = get_llm(temperature=0.0)
    prompt = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=ANSWER_USER_PROMPT.format(
            query=query,
            reasoning_notes=notes,
            context=context_str,
        )),
    ]

    try:
        llm_calls += 1
        resp = call_llm_with_retry(llm, prompt)
        answer_text = resp.content.strip()
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        answer_text = (
            "I encountered an error while synthesizing the clinical guideline answer. "
            "Please try again or consult the reference documents directly."
        )

    # Append official medical disclaimer
    final_answer = f"{answer_text}{MEDICAL_DISCLAIMER}"
    latency = time.time() - start_time

    return {
        "answer": final_answer,
        "llm_calls": llm_calls,
        "node_latencies": {**state.get("node_latencies", {}), "answer": latency},
    }


def out_of_scope_node(state: MedicalAssistantState) -> Dict[str, Any]:
    """
    Out of scope node: Returns polite refusal and scope guidance without LLM call.
    """
    start_time = time.time()
    return {
        "answer": OUT_OF_SCOPE_RESPONSE,
        "sources": [],
        "node_latencies": {**state.get("node_latencies", {}), "out_of_scope": time.time() - start_time},
    }

