"""Streamlit web interface for the AI Medical Assistant."""

import os
import json
import streamlit as st
from pathlib import Path

from src.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    INDEX_INFO_FILE,
    DATA_CHROMA_DIR,
    DOCUMENT_TITLES,
)
from src.graph import get_compiled_graph, run_query
from src.agents import clear_session_caches
from src.prompts import MEDICAL_DISCLAIMER

# Page configuration
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .route-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
    }
    .badge-qa { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .badge-emergency { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-scope { background-color: #FEF3C7; color: #B45309; border: 1px solid #FDE68A; }
    .source-box {
        background-color: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        border-radius: 0 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_graph():
    """Cache the compiled LangGraph workflow across sessions."""
    return get_compiled_graph()


def load_index_metadata():
    """Load metadata from index_info.json if available."""
    if INDEX_INFO_FILE.exists():
        try:
            with open(INDEX_INFO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    st.title("🩺 AI Medical Assistant")
    st.caption("Evidence-based clinical guidelines assistant powered by LangGraph, LangChain, and Chroma.")

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_query_stats" not in st.session_state:
        st.session_state.last_query_stats = None

    # Load vector store info
    index_info = load_index_metadata()

    # =====================================================================
    # Sidebar
    # =====================================================================
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This assistant answers medical questions strictly grounded in published "
            "clinical guidelines (WHO, ICMR STWs, and NTEP Guidelines). Every clinical claim includes "
            "an inline citation with page numbers."
        )

        st.divider()

        # API Key status
        active_key = os.getenv("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
        if not active_key:
            st.warning("⚠️ Google API Key not found in `.env`.")
            user_key = st.text_input("Enter Gemini API Key:", type="password", help="Required to run LLM queries.")
            if user_key:
                os.environ["GOOGLE_API_KEY"] = user_key
                st.success("API key loaded for this session.")
        else:
            st.success("🔑 Gemini API Key configured", icon="✅")

        st.caption(f"**Model:** `{os.getenv('GEMINI_MODEL', GEMINI_MODEL)}`")
        st.caption(f"**Embeddings:** `{index_info.get('embedding_model', EMBEDDING_MODEL) if index_info else EMBEDDING_MODEL}`")

        # Knowledge Base Info
        st.divider()
        st.subheader("📚 Ingested Knowledge Base")
        if index_info:
            st.markdown(f"**Total Documents:** {index_info.get('num_documents', 0)}")
            st.markdown(f"**Total Chunks:** {index_info.get('num_chunks', 0)}")
            st.markdown(f"**Pages Indexed:** {index_info.get('num_pages', 0)}")

            with st.expander("Loaded Documents"):
                for filename, title in DOCUMENT_TITLES.items():
                    st.markdown(f"- **{title}** (`{filename}`)")
        else:
            st.error("Vector store not found! Please run `python -m src.ingest --reset`.")

        # Observability Metrics
        st.divider()
        st.subheader("📊 Query Metrics")
        if st.session_state.last_query_stats:
            stats = st.session_state.last_query_stats
            st.metric("LLM Calls", stats.get("llm_calls", 0))
            latencies = stats.get("node_latencies", {})
            total_lat = sum(latencies.values())
            st.metric("Total Latency", f"{total_lat:.2f}s")
            with st.expander("Latency Breakdown"):
                for node, lat in latencies.items():
                    st.write(f"- `{node}`: {lat:.2f}s")
        else:
            st.caption("Submit a question to see real-time pipeline metrics.")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_query_stats = None
            clear_session_caches()
            st.rerun()

        st.caption(MEDICAL_DISCLAIMER)

    # Check for empty vector store
    if not index_info:
        st.error(
            "⚠️ **Knowledge Base Not Found!**\n\n"
            "The Chroma vector store is empty. Please open your terminal and run:\n"
            "```bash\npython -m src.ingest --reset\n```\n"
            "Then refresh this page."
        )
        return

    # =====================================================================
    # Chat History Rendering
    # =====================================================================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                route = msg.get("route", "document_qa")
                if route == "emergency":
                    st.markdown('<span class="route-badge badge-emergency">🚨 Emergency Response</span>', unsafe_allow_html=True)
                elif route == "out_of_scope":
                    st.markdown('<span class="route-badge badge-scope">ℹ️ Out of Scope</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="route-badge badge-qa">🩺 Clinical Document Q&A</span>', unsafe_allow_html=True)

                st.markdown(msg["content"])

                # Render sources
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📖 Sources ({len(sources)} cited)"):
                        for s in sources:
                            st.markdown(
                                f"<div class='source-box'>"
                                f"<strong>{s['title']}</strong> (Page {s['page']})<br/>"
                                f"<em>File:</em> <code>{s['file']}</code><br/>"
                                f"<small>{s['snippet']}</small>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown(msg["content"])

    # =====================================================================
    # Chat Input & Workflow Execution
    # =====================================================================
    user_prompt = st.chat_input("Ask a clinical guideline question (e.g., 'What are the first-line drugs for hypertension?')...")
    if user_prompt:
        # Check API key before making LLM query
        if not os.getenv("GOOGLE_API_KEY", "") and not GOOGLE_API_KEY:
            st.error("Please provide a valid `GOOGLE_API_KEY` in the sidebar or in `.env` to execute queries.")
            return

        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Build conversation history list
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        # Execute workflow with spinner
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query, searching clinical guidelines, and synthesizing answer..."):
                try:
                    result = run_query(user_prompt, chat_history=history)
                    answer = result.get("answer", "")
                    route = result.get("route", "document_qa")
                    sources = result.get("sources", [])

                    # Store metrics
                    st.session_state.last_query_stats = {
                        "llm_calls": result.get("llm_calls", 0),
                        "node_latencies": result.get("node_latencies", {}),
                    }

                    # Render response
                    if route == "emergency":
                        st.markdown('<span class="route-badge badge-emergency">🚨 Emergency Response</span>', unsafe_allow_html=True)
                    elif route == "out_of_scope":
                        st.markdown('<span class="route-badge badge-scope">ℹ️ Out of Scope</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="route-badge badge-qa">🩺 Clinical Document Q&A</span>', unsafe_allow_html=True)

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📖 Sources ({len(sources)} cited)"):
                            for s in sources:
                                st.markdown(
                                    f"<div class='source-box'>"
                                    f"<strong>{s['title']}</strong> (Page {s['page']})<br/>"
                                    f"<em>File:</em> <code>{s['file']}</code><br/>"
                                    f"<small>{s['snippet']}</small>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "route": route,
                        "sources": sources,
                    })

                except Exception as e:
                    err_msg = str(e)
                    if "rate limit" in err_msg.lower() or "429" in err_msg.lower():
                        st.error("⏳ **Gemini Rate Limit Reached**: The free-tier project limit was reached. Please wait 60 seconds and try again.")
                    elif "safety" in err_msg.lower() or "blocked" in err_msg.lower():
                        st.warning("🛡️ **Response Blocked by Safety Filters**: Gemini flagged this medical query. Please rephrase your question.")
                    else:
                        st.error(f"❌ **Error processing query:** {err_msg}")


if __name__ == "__main__":
    main()

