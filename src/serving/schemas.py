"""Compatibility re-export for serving code."""

from src.core.schemas import (
    AnswerResult,
    ChunkRecord,
    Citation,
    ParsedPage,
    ParsedTable,
    RetrievalCandidate,
)

__all__ = [
    "ParsedPage",
    "ParsedTable",
    "ChunkRecord",
    "RetrievalCandidate",
    "Citation",
    "AnswerResult",
]

