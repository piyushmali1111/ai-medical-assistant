# AI Medical Assistant Architecture & Design

This document details the architectural topology, state machine, and engineering trade-offs of the **AI Medical Assistant using LangGraph**.

---

## 1. High-Level System Architecture

The assistant implements an agentic workflow using **LangGraph** where an incoming query is classified by a triage router and handed off to specialized agents for retrieval, clinical reasoning, and grounded answer synthesis.

```mermaid
flowchart TD
    Start([User Query]) --> Router[Router Node\nTriage & Intent Classification]
    
    Router -->|emergency| Emergency[Emergency Node\nStatic Redirection - No LLM]
    Router -->|out_of_scope| OutOfScope[Out of Scope Node\nGuidance & Scope Refusal]
    Router -->|document_qa| Retriever[Retriever Node\nQuery Rewrite + Top-K Search]
    
    Retriever --> RelevanceCheck{Relevance OK?}
    RelevanceCheck -->|Yes| Reasoning[Reasoning Node\nClinical Extraction & Gap Analysis]
    RelevanceCheck -->|No| AnswerNotFound[Answer Node\nStrict Grounded Fallback]
    
    Reasoning --> Answer[Answer Node\nSynthesis with [Title, p.X] Citations]
    
    Emergency --> EndNode([Response + Disclaimer])
    OutOfScope --> EndNode
    AnswerNotFound --> EndNode
    Answer --> EndNode
```

---

## 2. LangGraph State Machine

The complete execution state is maintained in `MedicalAssistantState` (`src/state.py`):

| Field | Type | Description |
|---|---|---|
| `query` | `str` | Original user question |
| `chat_history` | `List[Dict[str, str]]` | Recent conversational context (up to 4 turns) |
| `route` | `str` | Routing branch: `document_qa`, `emergency`, or `out_of_scope` |
| `route_reason` | `Optional[str]` | Brief justification for the classification |
| `rewritten_query` | `Optional[str]` | Standalone search query optimized for vector lookup |
| `documents` | `List[Document]` | Retrieved chunks with source, title, and page metadata |
| `relevance_ok` | `bool` | Evaluation verdict on whether retrieved chunks answer the query |
| `retry_count` | `int` | Number of query reformulation attempts (max 1) |
| `reasoning_notes` | `Optional[str]` | Intermediate clinical extraction notes |
| `answer` | `str` | Final synthesized response with inline citations & disclaimer |
| `sources` | `List[Dict[str, Any]]` | Structured list of cited sources for UI inspection |
| `llm_calls` | `int` | Total LLM API calls consumed in the query |
| `node_latencies` | `Dict[str, float]` | Per-node execution time in seconds |

---

## 3. Engineering Decisions & Design Rationale

### 3.1 Why LangGraph Multi-Agent Architecture?
- **Deterministic Control**: Clinical safety requires hard guarantees. Urgent symptoms (chest pain, stroke signs) bypass generative models entirely via static code paths (`emergency` node), eliminating hallucination risk and zeroing out latency.
- **Separation of Concerns**: Decomposing search reformulation, semantic retrieval, clinical reasoning, and response generation into discrete nodes enables modular testing, targeted caching, and granular observability.

### 3.2 Chunking Strategy
- **Chunk Size: 800 characters, Overlap: 150 characters**.
- Selected via empirical validation: 800 characters corresponds to ~120-150 words—the typical length of a clinical recommendation, diagnostic threshold table row, or drug dosage specification in medical guidelines.
- Overlap of 150 characters prevents splitting critical sentence structures across chunk boundaries.

### 3.3 Embedding Model Selection
- Benchmarked three candidate models across 15 clinical benchmark questions:
  1. `BAAI/bge-small-en-v1.5` (384-dim): 100% Hit@5, 0.9222 MRR, 60.89 ms latency.
  2. `BAAI/bge-base-en-v1.5` (768-dim): 100% Hit@5, 0.9556 MRR, 119.54 ms latency.
  3. `all-MiniLM-L6-v2` (384-dim): 100% Hit@5, 0.8467 MRR, 25.72 ms latency.
- **Decision**: `BAAI/bge-small-en-v1.5` was chosen as the default for its optimal trade-off: 100% top-5 retrieval accuracy, high MRR, 3x faster ingestion than base, and lightweight local CPU footprint.

### 3.4 Hallucination Mitigation & Grounding Controls
1. **Low Temperature**: Set to `0.0` across router, grader, reasoning, and answer generation.
2. **Context-Only Prompts**: Strict prompt instructions prohibiting external assumptions.
3. **Mandatory Inline Citations**: Model must cite using exact chunk metadata tags (`[Title, p.X]`).
4. **Grounded 'Not Found' Fallback**: If retrieved chunks lack evidence, the assistant explicitly reports that the corpus does not cover the topic rather than guessing.
5. **Always-On Medical Disclaimer**: Attached to every output.

### 3.5 Rate-Limit & Quota Optimization (Gemini Free Tier)
- **Central Rate Limiter**: Enforces a configurable requests-per-minute ceiling (`RPM_LIMIT=15`).
- **Exponential Backoff**: 4 retry attempts with randomized jitter on transient 429/quota errors.
- **In-Memory Caching**: Router classifications and query rewrites are cached across turns in a session.
- **Emergency Bypass**: Zero LLM calls for emergency queries via heuristic shortcut.

