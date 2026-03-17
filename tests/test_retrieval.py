from __future__ import annotations

from pathlib import Path

import pytest

from rag_fin.indexing import RetrievalBuildConfig, build_retrieval_baseline
from rag_fin.retrieval.bm25 import bm25_retrieve
from rag_fin.retrieval.dense import build_embedding_model, dense_retrieve
from rag_fin.retrieval.formatting import format_retrieval_results
from rag_fin.utils.io import read_jsonl, write_jsonl


def _write_parsed_pages(parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        {
            "doc_id": "doc_fin_001",
            "title": "Synthetic Financial Report",
            "source_path": "data/raw_pdfs/synthetic_report.pdf",
            "page_num": 1,
            "text": "Revenue increased by 10 percent year over year.",
            "metadata": {"sector": "tech"},
            "parsing_warnings": [],
        },
        {
            "doc_id": "doc_fin_001",
            "title": "Synthetic Financial Report",
            "source_path": "data/raw_pdfs/synthetic_report.pdf",
            "page_num": 2,
            "text": "Operating margin improved by 2.1 percentage points.",
            "metadata": {"sector": "tech"},
            "parsing_warnings": [],
        },
        {
            "doc_id": "doc_policy_001",
            "title": "Synthetic Policy Brief",
            "source_path": "data/raw_pdfs/policy_brief.pdf",
            "page_num": 3,
            "text": "政策支持制造业升级与科技创新融资。",
            "metadata": {"domain": "policy"},
            "parsing_warnings": [],
        },
    ]
    write_jsonl(parsed_dir / "synthetic.pages.jsonl", pages)


@pytest.fixture()
def built_index(tmp_path: Path) -> Path:
    parsed_dir = tmp_path / "parsed"
    index_dir = tmp_path / "indexes"
    _write_parsed_pages(parsed_dir)

    config = RetrievalBuildConfig(
        embedding_model="mock",
        mock_embedding_dim=24,
        chunk_size=120,
        chunk_overlap=20,
        top_k=2,
        dense_top_k=2,
        bm25_top_k=2,
    )
    build_retrieval_baseline(parsed_dir=parsed_dir, index_dir=index_dir, config=config)
    return index_dir


def test_chunk_metadata_preservation(built_index: Path) -> None:
    chunks = read_jsonl(built_index / "chunks.jsonl")
    assert chunks
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert "doc_id" in metadata
        assert "title" in metadata
        assert "source_path" in metadata
        assert "page_start" in metadata
        assert "page_end" in metadata
        assert metadata["page_start"] <= metadata["page_end"]


def test_faiss_retrieval_wiring(built_index: Path) -> None:
    embedding_model = build_embedding_model(
        model_name="mock",
        mock_embedding_dim=24,
    )
    results = dense_retrieve(
        query="revenue growth",
        index_dir=built_index,
        embedding_model=embedding_model,
        top_k=2,
    )
    assert len(results) == 2
    assert all(item.retrieval_source == "dense" for item in results)


def test_bm25_retrieval_wiring(built_index: Path) -> None:
    results = bm25_retrieve(
        query="政策 科技 创新",
        index_dir=built_index,
        top_k=2,
    )
    assert len(results) == 2
    assert all(item.retrieval_source == "bm25" for item in results)
    assert any(item.metadata.doc_id == "doc_policy_001" for item in results)


def test_retrieval_result_formatting(built_index: Path) -> None:
    results = bm25_retrieve(
        query="operating margin",
        index_dir=built_index,
        top_k=2,
    )
    payload = format_retrieval_results(
        query="operating margin",
        results=results,
        retriever_name="bm25_rank_bm25",
    )
    assert payload["query"] == "operating margin"
    assert payload["retriever"] == "bm25_rank_bm25"
    assert payload["count"] == 2
    assert "citation" in payload["results"][0]
