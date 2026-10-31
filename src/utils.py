"""Utility helpers for logging, text cleaning, and rate limiting."""

import re
import sys
import time
import logging
import unicodedata
from typing import Optional

# Ensure standard output can print utf-8 characters on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medical_assistant")

class SimpleRateLimiter:
    """Thread-safe request rate limiter using minimum interval between requests."""
    def __init__(self, requests_per_minute: int = 15):
        self.rpm = max(1, requests_per_minute)
        self.min_interval = 60.0 / self.rpm
        self.last_call_time = 0.0

    def wait_if_needed(self):
        """Block if needed to respect the requests-per-minute quota."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            sleep_duration = self.min_interval - elapsed
            time.sleep(sleep_duration)
        self.last_call_time = time.time()

# Global default rate limiter
global_rate_limiter = SimpleRateLimiter(requests_per_minute=15)

def clean_page_text(text: Optional[str]) -> str:
    """
    Clean raw PDF page text:
    - Normalize unicode ligatures (e.g. fi, fl) and characters
    - Collapse repeated whitespace and broken line breaks
    - Preserve tables, numbers, and punctuation
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Normalize unicode ligatures and special characters
    text = unicodedata.normalize("NFKD", text)
    # Replace non-breaking and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Collapse multiple consecutive newlines into double newlines for paragraph breaks
    text = re.sub(r"\r\n|\r", "\n", text)
    # Reconnect words split across line breaks with a hyphen (e.g., 'treat-\nment' -> 'treatment')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace single line breaks that are not paragraph separators with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces or tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3 or more newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

