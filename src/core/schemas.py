"""Typed schema definitions used across ingestion, retrieval, and serving."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseRecord(BaseModel):
    """Shared model settings for all data records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ParsedPage(BaseRecord):
    """Represents one parsed PDF page."""

    doc_id: str
    file_name: str
    page_no: int
    text: str
    language: str = "zh"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseRecord):
    """Represents one table extracted from a page."""

    doc_id: str
    file_name: str
    page_no: int
    table_id: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    source_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseRecord):
    """Represents one chunk used for indexing and retrieval."""

    doc_id: str
    file_name: str
    page_no: int
    chunk_id: str
    chunk_type: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalCandidate(BaseRecord):
    """Represents one retrieved chunk candidate."""

    query_id: str
    query_text: str
    doc_id: str
    file_name: str
    page_no: int
    chunk_id: str
    chunk_type: str
    text: str
    rank: int
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseRecord):
    """Represents one citation attached to a generated answer."""

    doc_id: str
    file_name: str
    page_no: int
    chunk_id: str
    chunk_type: str
    quote: str | None = None
    score: float | None = None


class AnswerResult(BaseRecord):
    """Represents an answer grounded in retrieved evidence."""

    query_id: str
    query_text: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieval_candidates: list[RetrievalCandidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

