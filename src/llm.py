"""LLM and Embeddings initialization with rate-limiting, retries, and error handling."""

import os
import time
import random
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception

from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT,
    RPM_LIMIT,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    BGE_QUERY_INSTRUCTION,
)
from src.utils import SimpleRateLimiter, logger

# Central rate limiter
rate_limiter = SimpleRateLimiter(requests_per_minute=RPM_LIMIT)

class BGEHuggingFaceEmbeddings(Embeddings):
    """
    Wrapper for HuggingFace embeddings that automatically prepends
    the recommended BGE query instruction prefix when embedding queries.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
    ):
        self.model_name = model_name
        self.query_instruction = query_instruction
        self._hf_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._hf_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        # Only prepend instruction if model is BGE and prefix is not already present
        if self.query_instruction and "bge" in self.model_name.lower():
            if not text.startswith(self.query_instruction):
                text = f"{self.query_instruction}{text}"
        return self._hf_embeddings.embed_query(text)


def get_embeddings(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Embeddings:
    """
    Instantiate and return the configured embedding model.
    Defaults to local HuggingFace BAAI/bge-small-en-v1.5.
    """
    prov = (provider or EMBEDDING_PROVIDER).lower()
    m_name = model_name or EMBEDDING_MODEL

    if prov == "gemini":
        key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY must be set in .env to use Gemini embeddings."
            )
        return GoogleGenerativeAIEmbeddings(
            model=m_name if "models/" in m_name else f"models/{m_name}",
            google_api_key=key,
        )

    # Default to HuggingFace
    query_prefix = (
        BGE_QUERY_INSTRUCTION if "bge" in m_name.lower() else ""
    )
    return BGEHuggingFaceEmbeddings(
        model_name=m_name,
        query_instruction=query_prefix,
    )


def is_transient_error(exception: Exception) -> bool:
    """Identify if an exception is a rate limit or transient network error."""
    msg = str(exception).lower()
    transient_indicators = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "deadline_exceeded",
        "timeout",
        "temporarily unavailable",
        "503",
    ]
    return any(ind in msg for ind in transient_indicators)


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """
    Single central factory for Gemini Chat LLM instances.
    Reads GOOGLE_API_KEY and GEMINI_MODEL from config/environment.
    """
    key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "") or "AIzaSy_PLACEHOLDER_KEY"
    chosen_model = model or GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    chosen_temp = TEMPERATURE if temperature is None else temperature
    chosen_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    chosen_timeout = REQUEST_TIMEOUT if timeout is None else timeout

    return ChatGoogleGenerativeAI(
        model=chosen_model,
        google_api_key=key,
        temperature=chosen_temp,
        max_output_tokens=chosen_tokens,
        timeout=chosen_timeout,
        max_retries=0, # Retries handled explicitly in our wrapper
    )


class LLMCallTracker:
    """Helper to track call metrics and handle retries."""
    def __init__(self):
        self.call_count = 0
        self.total_latency = 0.0

    def reset(self):
        self.call_count = 0
        self.total_latency = 0.0


def call_llm_with_retry(runnable: Any, input_data: Any) -> Any:
    """
    Executes an LLM or runnable invocation with:
    - Central rate-limiter throttle
    - Exponential backoff with jitter on 429/quota/transient errors (max 4 attempts)
    - Safety filter / blocked response detection
    - Friendly error message if rate limit persists
    """
    attempts = 0
    max_attempts = 4

    while attempts < max_attempts:
        attempts += 1
        rate_limiter.wait_if_needed()
        start_time = time.time()
        try:
            response = runnable.invoke(input_data)
            latency = time.time() - start_time
            logger.debug("LLM invocation succeeded in %.2fs (attempt %d)", latency, attempts)

            # Check for safety filter blocks or empty responses
            if hasattr(response, "response_metadata"):
                finish_reason = response.response_metadata.get("finish_reason", "")
                if finish_reason in ["SAFETY", "BLOCK", "RECITATION"]:
                    logger.warning(
                        "Gemini blocked response due to safety filter: %s", finish_reason
                    )
                    raise ValueError(
                        f"Response was blocked by safety filters (reason: {finish_reason}). "
                        "Medical content can sometimes trigger sensitive topic filters."
                    )

            if hasattr(response, "content") and not response.content and not getattr(response, "parsed", None):
                logger.warning("Gemini returned an empty response on attempt %d", attempts)

            return response

        except Exception as e:
            latency = time.time() - start_time
            if is_transient_error(e):
                logger.warning(
                    "Transient or rate-limit error (attempt %d/%d): %s",
                    attempts,
                    max_attempts,
                    e,
                )
                if attempts < max_attempts:
                    # Exponential backoff with jitter
                    sleep_time = (2 ** attempts) + random.uniform(0.5, 1.5)
                    time.sleep(sleep_time)
                    continue
                else:
                    raise RuntimeError(
                        "Gemini rate limit reached or service temporarily unavailable. "
                        "Please wait a minute before making another request."
                    ) from e
            else:
                # Non-transient error, re-raise immediately
                logger.error("LLM call failed with error: %s", e)
                raise e

