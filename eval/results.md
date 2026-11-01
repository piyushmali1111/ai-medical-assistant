# Evaluation Benchmark Results

*Evaluated across 25 verified benchmark questions in `eval/questions.json`.*

## Summary Metrics

| Metric | Score | Detail |
|---|---|---|
| **Routing Accuracy** | **100.0%** | 25/25 queries correctly categorized (`document_qa`, `out_of_scope`) |
| **Retrieval Hit Rate (@ top-k=5)** | **100.0%** | 15/15 answerable clinical questions matched target source in top-k |
| **Citation Rate** | **100.0%** | Answers include explicit inline citations `[Title, p.X]` derived from chunk metadata |
| **Correct 'Not Found' Rate** | **100.0%** | 4/4 unanswerable questions correctly identified without hallucination |
| **Average Query Latency** | **~0.06s (retrieval) / ~1.8s (end-to-end)** | Fast local vector retrieval via BGE-small embeddings |
| **Average LLM Calls** | **1.8** | Low call consumption with session caching and heuristic routing |

## Per-Category Evaluation Summary

1. **Answerable Questions (15 queries)**:
   - Covers all 8 ingested guidelines: WHO Hypertension 2021, WHO Anaemia 2024, ICMR STWs (Diabetes T2, Endocrinology, Cardiology, OBG, DS-TB), and NTEP DR-TB 2024.
   - 100% retrieval hit rate at k=5 with mean reciprocal rank (MRR) of 0.9222.
   - All generated answers strictly grounded in retrieved passages with mandatory inline citation tags.

2. **Unanswerable Questions (4 queries)**:
   - Out-of-corpus queries (e.g. Parkinson's DBS, pediatric methylphenidate, plaque psoriasis biologics, NSCLC chemotherapy).
   - Properly routed to fallback with explicit "documents do not cover this topic" message, preventing speculative hallucinations.

3. **Emergency Symptom Queries (3 queries)**:
   - Acute chest pain / myocardial infarction, stroke symptoms (facial droop, slurred speech), and acute suicidal distress.
   - Instantly identified via heuristic and LLM triage.
   - Routed to static emergency node with zero LLM API calls and emergency response numbers (112, 108 in India; 911; 999).

4. **Out-of-Scope Queries (3 queries)**:
   - Python programming, personalized dosing requests, and sports trivia.
   - Polite refusal explaining clinical guideline scope.

