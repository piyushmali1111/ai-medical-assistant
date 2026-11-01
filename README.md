# AI Medical Assistant using LangGraph

An evidence-based clinical guidelines assistant built with **LangGraph**, **LangChain**, **Chroma**, and **Google Gemini API**. Answers clinical questions with grounded responses and precise inline source citations (`[Title, p.X]`) derived from official medical guidelines (WHO, ICMR STWs, NTEP).

---

## 🌟 Key Features

1. **Multi-Agent Orchestration with LangGraph**:
   - **Triage Router**: Dynamically categorizes queries into `document_qa`, `emergency`, or `out_of_scope`.
   - **Zero-Latency Emergency Path**: Critical symptoms (chest pain, stroke signs, breathing difficulty) trigger immediate emergency hotline guidance (112 / 108 in India; 911; 999) without consuming LLM API calls.
   - **Contextual Retrieval & Query Rewriting**: Resolves pronouns and conversational follow-ups into standalone medical search queries.
   - **Clinical Reasoning & Grounded Synthesis**: Synthesizes verified facts with mandatory inline citations.

2. **RAG Pipeline with Local Embeddings**:
   - Built on **ChromaDB** with local **HuggingFace** embeddings (`BAAI/bge-small-en-v1.5`).
   - Zero quota consumption and complete data privacy for document indexing and semantic retrieval.

3. **LangSmith Observability**:
   - Full end-to-end tracing enabled via environment variables.
   - Inspect router decisions, rewritten queries, retrieved chunks, reasoning notes, token usage, and latencies.

4. **Production Streamlit Chat Interface**:
   - Real-time conversational interface with multi-turn support.
   - Per-response route badges and expandable **Sources** panel displaying cited titles, pages, and text snippets.
   - Observability sidebar showing active model, indexed document stats, and per-query LLM call counts.

---

## 🏗️ System Architecture

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

## 🔬 Benchmark & Evaluation Results

### 1. Pipeline Evaluation (`eval/results.md`)
Evaluated across **25 verified clinical benchmark questions** (`eval/questions.json`):

| Metric | Score | Detail |
|---|---|---|
| **Routing Accuracy** | **100.0%** | 25/25 queries correctly categorized (`document_qa`, `emergency`, `out_of_scope`) |
| **Retrieval Hit Rate (@ top-k=5)** | **100.0%** | 15/15 answerable clinical questions matched target source in top-k |
| **Citation Rate** | **100.0%** | Answers include explicit inline citations `[Title, p.X]` derived from chunk metadata |
| **Correct 'Not Found' Rate** | **100.0%** | 4/4 unanswerable questions correctly identified without hallucination |
| **Average Query Latency** | **~0.06s (retrieval) / ~1.8s (end-to-end)** | Fast local vector retrieval via BGE-small embeddings |
| **Average LLM Calls** | **1.8** | Low call consumption with session caching and heuristic routing |

### 2. Embedding Model Comparison (`eval/embedding_comparison.md`)
Benchmarked locally across all 8 ingested clinical documents (1,207 chunks) and 15 answerable questions:

| Model | Dim | Ingestion Time | Avg Latency | Hit@3 | Hit@5 | Hit@8 | MRR |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` *(Default)* | 384 | 195.34s | 60.89 ms | **100.0%** | **100.0%** | 100.0% | **0.9222** |
| `BAAI/bge-base-en-v1.5` | 768 | 570.38s | 119.54 ms | 100.0% | 100.0% | 100.0% | **0.9556** |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 84.55s | 25.72 ms | 93.3% | 100.0% | 100.0% | **0.8467** |

**Selection Decision:** `BAAI/bge-small-en-v1.5` was chosen as the default production model. It achieves **100.0% Hit@5** and an exceptional **0.9222 MRR** while requiring **3x less ingestion time** and **half the query latency and memory footprint** of the base model.

---

## 📚 Knowledge-Base Documents

The assistant is indexed over the following official clinical guidelines:

| Short Name | Document Title & Publisher | Official URL |
|---|---|---|
| `who_hypertension_2021.pdf` | WHO: Pharmacological treatment of hypertension in adults | [WHO Publication](https://www.who.int/publications/i/item/9789240033986) |
| `who_anaemia_hb_cutoffs_2024.pdf` | WHO: Haemoglobin cutoffs to define anaemia | [WHO Publication](https://www.who.int/publications/i/item/9789240088542) |
| `icmr_stw_diabetes_t2.pdf` | ICMR Standard Treatment Workflow: Type 2 Diabetes | [ICMR STW](https://icmr.gov.in/standard-treatment-workflows-stws) |
| `icmr_stw_endocrinology.pdf` | ICMR Standard Treatment Workflow: Endocrinology | [ICMR STW](https://icmr.gov.in/standard-treatment-workflows-stws) |
| `icmr_stw_cardiology.pdf` | ICMR Standard Treatment Workflow: Cardiology | [ICMR STW](https://icmr.gov.in/standard-treatment-workflows-stws) |
| `icmr_stw_obg.pdf` | ICMR Standard Treatment Workflow: Obstetrics and Gynaecology | [ICMR STW](https://icmr.gov.in/standard-treatment-workflows-stws) |
| `icmr_stw_ds_tb_ntep.pdf` | ICMR Standard Treatment Workflow: Drug-sensitive TB (NTEP) | [ICMR STW](https://icmr.gov.in/standard-treatment-workflows-stws) |
| `ntep_dr_tb_guidelines_2024.pdf` | NTEP: National Guidelines for Management of Drug-Resistant TB | [Central TB Division](https://tbcindia.mohfw.gov.in) |

*Licensing & Data Notice: The WHO guideline is distributed under CC BY-NC-SA 3.0 IGO. ICMR workflows are used for local clinical reference and are not redistributed in this repository.*

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.11+
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))
- Optional: LangSmith Account ([smith.langchain.com](https://smith.langchain.com/))

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/piyushmali1111/ai-medical-assistant.git
cd ai-medical-assistant

# Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API keys:
```bash
cp .env.example .env
```
Key configuration settings in `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=ai-medical-assistant
```
*(Default Gemini model: `gemini-2.5-flash`, verified October 2026).*

### 4. Download Guidelines & Ingest Knowledge Base
Download the PDFs into `data/raw/` or run the automated downloader:
```bash
python scripts/download_docs.py
```
Build the persistent Chroma vector store:
```bash
python -m src.ingest --reset
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🔍 Debugging with LangSmith

LangSmith tracing activates automatically when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are provided in `.env`:
- **Router Node Trace**: Inspect the structured Pydantic output (`RouterDecision`) and classification reason.
- **Retriever Node Trace**: Inspect the rewritten standalone query, similarity scores, and retrieved document chunks.
- **Reasoning Node Trace**: Verify factual extraction and gap analysis before answer synthesis.
- **Latency & Token Telemetry**: Monitor exact token consumption and millisecond latency per node.

Screenshots can be stored in [`docs/img/`](docs/img/).

---

## ⚡ Gemini Free-Tier Optimization & Rate Limits

1. **Per-Project Quotas**: Rate limits apply across your Google AI Studio project, not per API key.
2. **Exponential Backoff**: Built-in 4-attempt retry wrapper with randomized jitter for transient 429 errors.
3. **Emergency Bypass**: Emergency symptoms are filtered via heuristic rules and static templates, consuming 0 tokens and 0 API calls.
4. **Session Caching**: Router decisions and rewritten queries are cached in memory within each session.
5. **Data Privacy Advisory**: On Google's free tier, inputs may be used to train and refine models. **Do not input identifiable personal health data (PHI/PII).**

---

## 🧪 Testing

Run unit and integration tests (with LLM calls mocked):
```bash
pytest -v
```

Run manual 5-query retrieval test:
```bash
python scripts/test_retrieval_5.py
```

Run embedding benchmark:
```bash
python -m eval.compare_embeddings
```

---

## ⚖️ Medical Disclaimer

> **IMPORTANT:** This application is strictly intended for **educational, research, and reference purposes**. It is not a certified medical device and does not provide clinical diagnosis, medical advice, or personalized treatment plans. Always consult a licensed healthcare professional for medical decisions. In case of an emergency, immediately contact local emergency services (112 / 108 in India; 911 in North America; 999 in the UK).

