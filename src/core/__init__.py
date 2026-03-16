"""Core domain schemas and abstractions."""

from .schemas import (
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

