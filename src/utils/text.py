"""Small text helper functions for preprocessing."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def safe_truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, reserving room for suffix when needed."""
    if max_length < len(suffix):
        raise ValueError("max_length must be >= suffix length")
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def contains_cjk(text: str) -> bool:
    """Return True if text contains CJK Unified Ideographs."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

