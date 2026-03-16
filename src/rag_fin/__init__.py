"""finRAG package scaffold."""

from rag_fin.schemas import (
    ChunkMetadata,
    EvaluationSample,
    ParsedDocumentArtifact,
    ParsedPageRecord,
    RetrievalResult,
)

__all__ = [
    "ParsedPageRecord",
    "ParsedDocumentArtifact",
    "ChunkMetadata",
    "RetrievalResult",
    "EvaluationSample",
]
