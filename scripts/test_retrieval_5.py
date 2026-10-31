"""Verification script for Phase 1 Gate: 5 manual retrieval queries."""

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.retrieval import query_vector_store, format_citation

queries = [
    "What are the recommended first line drug classes for hypertension?",
    "What is the haemoglobin cutoff for defining anaemia in pregnant women?",
    "What are the diagnostic criteria and HbA1c threshold for type 2 diabetes?",
    "What is the regimen for drug sensitive pulmonary tuberculosis under NTEP?",
    "How should acute chest pain or myocardial infarction be managed in cardiology?",
]

def main():
    print("Testing 5 manual queries against Chroma vector store...\n")
    for idx, q in enumerate(queries, start=1):
        print(f"=== Query {idx}: {q} ===")
        docs = query_vector_store(q, k=3)
        for rank, d in enumerate(docs, start=1):
            citation = format_citation(d.metadata)
            source = d.metadata.get("source")
            page = d.metadata.get("page")
            snippet = d.page_content[:180].strip().replace("\n", " ")
            print(f"  [{rank}] {citation} (File: {source}, Page {page})")
            print(f"       Snippet: {snippet}...")
        print()

if __name__ == "__main__":
    main()
