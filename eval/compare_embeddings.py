"""Embedding model benchmark and comparison across local HuggingFace models."""

import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_chroma import Chroma
from src.config import DATA_RAW_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest import load_all_raw_documents, split_documents
from src.llm import get_embeddings
from src.utils import logger

MODELS_TO_BENCHMARK = [
    {
        "name": "BAAI/bge-small-en-v1.5",
        "provider": "huggingface",
        "slug": "bge_small",
        "description": "BGE Small v1.5 (384-dim, fast, lightweight)",
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "provider": "huggingface",
        "slug": "bge_base",
        "description": "BGE Base v1.5 (768-dim, higher capacity)",
    },
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "provider": "huggingface",
        "slug": "minilm_l6",
        "description": "all-MiniLM-L6-v2 (384-dim, standard sentence transformer)",
    },
]


def evaluate_model(
    model_cfg: Dict[str, Any],
    chunks: List[Any],
    answerable_questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build index for the model, evaluate retrieval hit rate @ k (3, 5, 8) and MRR.
    """
    model_name = model_cfg["name"]
    slug = model_cfg["slug"]
    persist_dir = DATA_DIR / f"chroma_bench_{slug}"
    shutil.rmtree(persist_dir, ignore_errors=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating: {model_name}...")
    embed_fn = get_embeddings(provider=model_cfg["provider"], model_name=model_name)

    # 1. Ingestion Benchmark
    t0 = time.time()
    vector_store = Chroma(
        collection_name=f"bench_{slug}",
        embedding_function=embed_fn,
        persist_directory=str(persist_dir),
    )

    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c.metadata["chunk_id"] for c in batch]
        vector_store.add_documents(batch, ids=ids)

    ingest_time = time.time() - t0
    print(f"  -> Ingested {len(chunks)} chunks in {ingest_time:.2f}s")

    # Get sample embedding dimension
    sample_emb = embed_fn.embed_query("test query")
    emb_dim = len(sample_emb)

    # 2. Retrieval Evaluation
    total_q = len(answerable_questions)
    hits_k3 = 0
    hits_k5 = 0
    hits_k8 = 0
    reciprocal_ranks = []
    latencies = []

    for q_item in answerable_questions:
        q_text = q_item["question"]
        expected_src = q_item["expected_source"]

        t_start = time.time()
        retrieved_docs = vector_store.similarity_search(q_text, k=8)
        latencies.append(time.time() - t_start)

        retrieved_sources = [d.metadata.get("source") for d in retrieved_docs]

        # Hit rate checks
        if expected_src in retrieved_sources[:3]:
            hits_k3 += 1
        if expected_src in retrieved_sources[:5]:
            hits_k5 += 1
        if expected_src in retrieved_sources[:8]:
            hits_k8 += 1

        # MRR calculation
        if expected_src in retrieved_sources:
            rank = retrieved_sources.index(expected_src) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    # Cleanup benchmark store to save disk space
    shutil.rmtree(persist_dir, ignore_errors=True)

    avg_latency = (sum(latencies) / len(latencies)) * 1000.0  # ms
    mrr = sum(reciprocal_ranks) / total_q

    results = {
        "model_name": model_name,
        "description": model_cfg["description"],
        "dimension": emb_dim,
        "ingest_time_s": round(ingest_time, 2),
        "avg_query_latency_ms": round(avg_latency, 2),
        "hit_rate_k3": round(hits_k3 / total_q * 100, 1),
        "hit_rate_k5": round(hits_k5 / total_q * 100, 1),
        "hit_rate_k8": round(hits_k8 / total_q * 100, 1),
        "mrr": round(mrr, 4),
        "total_evaluated": total_q,
    }
    print(f"  -> Dimension: {emb_dim} | Hit@5: {results['hit_rate_k5']}% | MRR: {mrr:.4f} | Latency: {avg_latency:.1f}ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark local embedding models.")
    parser.add_argument("--gemini", action="store_true", help="Include Gemini embeddings in benchmark")
    args = parser.parse_args()

    questions_path = Path(__file__).resolve().parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answerable = [q for q in questions if q.get("category") == "answerable" and q.get("expected_source")]
    print(f"Loaded {len(answerable)} answerable benchmark questions from {questions_path.name}")

    # Load and split documents once
    page_docs, _ = load_all_raw_documents()
    chunks = split_documents(page_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Loaded {len(page_docs)} pages -> {len(chunks)} chunks.")

    models = list(MODELS_TO_BENCHMARK)
    if args.gemini:
        models.append({
            "name": "text-embedding-004",
            "provider": "gemini",
            "slug": "gemini_text_004",
            "description": "Google Gemini text-embedding-004 (Cloud)",
        })

    results = []
    for m in models:
        res = evaluate_model(m, chunks, answerable)
        results.append(res)

    # Format Markdown Report
    output_path = Path(__file__).resolve().parent / "embedding_comparison.md"
    md_content = "# Embedding Model Comparison & Benchmark Report\n\n"
    md_content += f"*Generated on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*\n\n"
    md_content += f"Evaluated across **{len(answerable)} verified clinical benchmark questions** against **{len(chunks)} chunks** from 8 clinical guideline documents.\n\n"
    md_content += "| Model | Dim | Ingestion Time | Avg Latency | Hit@3 | Hit@5 | Hit@8 | MRR |\n"
    md_content += "|---|---|---|---|---|---|---|---|\n"

    for r in results:
        md_content += (
            f"| `{r['model_name']}` | {r['dimension']} | {r['ingest_time_s']}s | "
            f"{r['avg_query_latency_ms']} ms | {r['hit_rate_k3']}% | {r['hit_rate_k5']}% | "
            f"{r['hit_rate_k8']}% | **{r['mrr']}** |\n"
        )

    best_model = max(results, key=lambda x: (x["hit_rate_k5"], x["mrr"], -x["avg_query_latency_ms"]))
    md_content += "\n## Benchmark Analysis & Decision\n\n"
    md_content += f"- **Top Performer / Recommended Default:** `{best_model['model_name']}`\n"
    md_content += (
        f"- **Rationale:** Demonstrates high retrieval accuracy (Hit@5: {best_model['hit_rate_k5']}%, MRR: {best_model['mrr']}) "
        f"with fast query retrieval ({best_model['avg_query_latency_ms']} ms/query) and local CPU execution with zero quota usage.\n"
        "- **Privacy & Offline Independence:** Local HuggingFace sentence-transformers ensure reproducible offline execution without API rate limits or privacy leaks.\n"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nBenchmark completed successfully! Report written to: {output_path}")


if __name__ == "__main__":
    main()

