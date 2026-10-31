"""Configuration settings for AI Medical Assistant."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_CHROMA_DIR = DATA_DIR / "chroma"
INDEX_INFO_FILE = DATA_CHROMA_DIR / "index_info.json"

# LLM / Gemini Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Default model name is read from environment variable GEMINI_MODEL
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RPM_LIMIT = int(os.getenv("RPM_LIMIT", "15"))

# Embeddings Settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Vector DB Settings
CHROMA_COLLECTION_NAME = "medical_docs"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Feature Flags
ENABLE_RELEVANCE_GRADER = os.getenv("ENABLE_RELEVANCE_GRADER", "true").lower() in ("true", "1", "yes")
ENABLE_SEPARATE_REASONING = os.getenv("ENABLE_SEPARATE_REASONING", "true").lower() in ("true", "1", "yes")

# Known human-readable document titles
DOCUMENT_TITLES = {
    "who_hypertension_2021.pdf": "WHO Guidelines on Pharmacological Treatment of Hypertension in Adults",
    "who_anaemia_hb_cutoffs_2024.pdf": "WHO Haemoglobin Cutoffs to Define Anaemia",
    "icmr_stw_diabetes_t2.pdf": "ICMR STW: Type 2 Diabetes Mellitus",
    "icmr_stw_endocrinology.pdf": "ICMR STW: Endocrinology",
    "icmr_stw_cardiology.pdf": "ICMR STW: Cardiology",
    "icmr_stw_obg.pdf": "ICMR STW: Obstetrics and Gynaecology",
    "icmr_stw_ds_tb_ntep.pdf": "ICMR STW: Drug-Sensitive TB under NTEP",
    "ntep_dr_tb_guidelines_2024.pdf": "NTEP Guidelines for Management of Drug-Resistant TB",
}

def get_document_title(filename: str) -> str:
    """Return a human-readable title for a source filename."""
    if filename in DOCUMENT_TITLES:
        return DOCUMENT_TITLES[filename]
    # Fallback title derived from filename
    clean_name = Path(filename).stem.replace("_", " ").replace("-", " ")
    return clean_name.title()

