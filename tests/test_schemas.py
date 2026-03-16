import pytest
from pydantic import ValidationError

from rag_fin.schemas import (
    ChunkMetadata,
    EvaluationSample,
    ParsedPageRecord,
    RetrievalResult,
)


def test_parsed_page_record_roundtrip() -> None:
    record = ParsedPageRecord(
        doc_id="doc_001",
        title="Synthetic Title",
        source_path="data/raw_pdfs/doc_001.pdf",
        page_num=1,
        text="Sample page text",
        metadata={"lang": "zh"},
    )
    assert record.page_num == 1
    assert record.metadata["lang"] == "zh"


def test_chunk_metadata_rejects_invalid_page_range() -> None:
    with pytest.raises(ValidationError):
        ChunkMetadata(
            chunk_id="chunk_001",
            doc_id="doc_001",
            title="Synthetic Title",
            source_path="data/raw_pdfs/doc_001.pdf",
            page_start=3,
            page_end=2,
        )


def test_retrieval_result_contains_chunk_metadata() -> None:
    chunk_meta = ChunkMetadata(
        chunk_id="chunk_001",
        doc_id="doc_001",
        title="Synthetic Title",
        source_path="data/raw_pdfs/doc_001.pdf",
        page_start=1,
        page_end=1,
    )
    result = RetrievalResult(
        chunk_id="chunk_001",
        text="Retrieved chunk text",
        score=0.87,
        retrieval_source="dense",
        metadata=chunk_meta,
    )
    assert result.metadata.doc_id == "doc_001"


def test_evaluation_sample_supports_refusal_case() -> None:
    sample = EvaluationSample(
        sample_id="eval_001",
        question="Unknown question",
        expected_answer=None,
        expected_citations=[],
        unanswerable=True,
    )
    assert sample.unanswerable is True
