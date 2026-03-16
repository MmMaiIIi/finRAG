"""Tests for core schema serialization behavior."""

from src.core.schemas import (
    AnswerResult,
    Citation,
    ChunkRecord,
    ParsedPage,
    ParsedTable,
    RetrievalCandidate,
)


def test_page_and_table_round_trip() -> None:
    page = ParsedPage(
        doc_id="doc-001",
        file_name="report.pdf",
        page_no=3,
        text="Revenue grew year over year in 2025.",
        metadata={"source": "unit-test"},
    )
    table = ParsedTable(
        doc_id="doc-001",
        file_name="report.pdf",
        page_no=3,
        table_id="t1",
        headers=["item", "amount"],
        rows=[["revenue", "100"]],
    )

    page_rt = ParsedPage.model_validate(page.model_dump())
    table_rt = ParsedTable.model_validate(table.model_dump())

    assert page_rt == page
    assert table_rt == table


def test_answer_result_round_trip_with_nested_objects() -> None:
    chunk = ChunkRecord(
        doc_id="doc-001",
        file_name="report.pdf",
        page_no=3,
        chunk_id="c-3-1",
        chunk_type="text",
        text="Revenue growth mainly came from core business.",
        token_count=18,
    )
    candidate = RetrievalCandidate(
        query_id="q1",
        query_text="What drove revenue growth?",
        doc_id=chunk.doc_id,
        file_name=chunk.file_name,
        page_no=chunk.page_no,
        chunk_id=chunk.chunk_id,
        chunk_type=chunk.chunk_type,
        text=chunk.text,
        rank=1,
        score=0.91,
    )
    citation = Citation(
        doc_id=chunk.doc_id,
        file_name=chunk.file_name,
        page_no=chunk.page_no,
        chunk_id=chunk.chunk_id,
        chunk_type=chunk.chunk_type,
        quote="Revenue growth mainly came from core business.",
        score=0.91,
    )
    answer = AnswerResult(
        query_id="q1",
        query_text="What drove revenue growth?",
        answer="The document states growth came from core business.",
        citations=[citation],
        retrieval_candidates=[candidate],
    )

    restored = AnswerResult.model_validate_json(answer.model_dump_json())

    assert restored == answer
    assert restored.citations[0].chunk_id == "c-3-1"
    assert restored.retrieval_candidates[0].rank == 1
