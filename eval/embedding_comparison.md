# Embedding Model Comparison & Benchmark Report

*Generated on 2026-10-31 07:53:16 UTC*

Evaluated across **15 verified clinical benchmark questions** against **1207 chunks** from 8 clinical guideline documents.

| Model | Dim | Ingestion Time | Avg Latency | Hit@3 | Hit@5 | Hit@8 | MRR |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 195.34s | 60.89 ms | 100.0% | 100.0% | 100.0% | **0.9222** |
| `BAAI/bge-base-en-v1.5` | 768 | 570.38s | 119.54 ms | 100.0% | 100.0% | 100.0% | **0.9556** |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 84.55s | 25.72 ms | 93.3% | 100.0% | 100.0% | **0.8467** |

## Benchmark Analysis & Decision

- **Top Performer / Recommended Default:** `BAAI/bge-base-en-v1.5`
- **Rationale:** Demonstrates high retrieval accuracy (Hit@5: 100.0%, MRR: 0.9556) with fast query retrieval (119.54 ms/query) and local CPU execution with zero quota usage.
- **Privacy & Offline Independence:** Local HuggingFace sentence-transformers ensure reproducible offline execution without API rate limits or privacy leaks.
