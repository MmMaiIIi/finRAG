from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ParsedPageRecord(BaseModel):
    """Parsed page-level record with provenance preserved."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    source_path: str
    page_num: int = Field(ge=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Chunk metadata used across indexing and retrieval."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_page_range(self) -> "ChunkMetadata":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class RetrievalResult(BaseModel):
    """Retrieval output item with score and provenance."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str
    score: float
    retrieval_source: str = Field(
        description="Source tag such as dense, bm25, hybrid, or rerank."
    )
    metadata: ChunkMetadata


class CitationRef(BaseModel):
    """Minimal citation target for evaluation references."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = None
    title: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_identity_and_range(self) -> "CitationRef":
        if not self.doc_id and not self.title:
            raise ValueError("either doc_id or title must be provided")
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class EvaluationSample(BaseModel):
    """Offline evaluation sample for QA and refusal checks."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str
    expected_answer: str | None = None
    expected_citations: list[CitationRef] = Field(default_factory=list)
    unanswerable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
