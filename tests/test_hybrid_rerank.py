from __future__ import annotations

from pathlib import Path

from rag_fin.indexing import RetrievalBuildConfig, build_retrieval_baseline
from rag_fin.retrieval.runner import run_retrieval
from rag_fin.rerank import build_reranker, rerank_candidates
from rag_fin.schemas import ChunkMetadata, RetrievalResult
from rag_fin.utils.io import write_jsonl


def _write_parsed_pages(parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        {
            "doc_id": "doc_a",
            "title": "Alpha Report",
            "source_path": "data/raw_pdfs/a.pdf",
            "page_num": 1,
            "text": "Revenue growth remained stable with better margin.",
            "metadata": {},
            "parsing_warnings": [],
        },
        {
            "doc_id": "doc_b",
            "title": "Beta Policy",
            "source_path": "data/raw_pdfs/b.pdf",
            "page_num": 2,
            "text": "Manufacturing support policy improves financing.",
            "metadata": {},
            "parsing_warnings": [],
        },
    ]
    write_jsonl(parsed_dir / "hybrid.pages.jsonl", pages)


def _build_mock_index(tmp_path: Path) -> Path:
    parsed_dir = tmp_path / "parsed"
    index_dir = tmp_path / "indexes"
    _write_parsed_pages(parsed_dir)
    cfg = RetrievalBuildConfig(
        embedding_model="mock",
        reranker_model="mock",
        mock_embedding_dim=16,
        chunk_size=120,
        chunk_overlap=20,
        dense_top_k=2,
        bm25_top_k=2,
        fused_top_n=2,
        rerank_top_n=2,
    )
    build_retrieval_baseline(parsed_dir=parsed_dir, index_dir=index_dir, config=cfg)
    return index_dir


def test_fusion_output_shape(tmp_path: Path) -> None:
    index_dir = _build_mock_index(tmp_path)
    payload = run_retrieval(
        query="manufacturing financing support",
        index_dir=index_dir,
        mode="hybrid",
        embedding_model_name="mock",
        embedding_device="cpu",
        mock_embedding_dim=16,
        dense_top_k=2,
        bm25_top_k=2,
        fused_top_n=2,
        rerank_top_n=2,
        fusion_strategy="rrf",
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        reranker_model="mock",
        reranker_backend="auto",
        reranker_use_fp16=False,
    )
    assert "fused" in payload
    assert payload["fused"]["retriever"] == "hybrid_rrf"
    assert isinstance(payload["fused"]["results"], list)
    if payload["fused"]["results"]:
        assert "debug" in payload["fused"]["results"][0]


def test_rerank_pipeline_wiring() -> None:
    candidates = [
        RetrievalResult(
            chunk_id="c1",
            text="General macro summary.",
            score=0.2,
            retrieval_source="hybrid_rrf",
            metadata=ChunkMetadata(
                chunk_id="c1",
                doc_id="d1",
                title="Doc1",
                source_path="a.pdf",
                page_start=1,
                page_end=1,
                metadata={},
            ),
        ),
        RetrievalResult(
            chunk_id="c2",
            text="Manufacturing support and financing policy details.",
            score=0.1,
            retrieval_source="hybrid_rrf",
            metadata=ChunkMetadata(
                chunk_id="c2",
                doc_id="d2",
                title="Doc2",
                source_path="b.pdf",
                page_start=2,
                page_end=2,
                metadata={},
            ),
        ),
    ]
    reranker = build_reranker(model_name="mock", backend="auto")
    reranked = rerank_candidates(
        query="manufacturing financing support",
        candidates=candidates,
        reranker=reranker,
        rerank_top_n=2,
    )
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].retrieval_source == "rerank"


def test_metadata_retention_after_reranking(tmp_path: Path) -> None:
    index_dir = _build_mock_index(tmp_path)
    payload = run_retrieval(
        query="revenue margin",
        index_dir=index_dir,
        mode="hybrid",
        embedding_model_name="mock",
        embedding_device="cpu",
        mock_embedding_dim=16,
        dense_top_k=2,
        bm25_top_k=2,
        fused_top_n=2,
        rerank_top_n=2,
        fusion_strategy="rrf",
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
        reranker_model="mock",
        reranker_backend="auto",
        reranker_use_fp16=False,
    )
    results = payload["reranked"]["results"]
    assert results
    first = results[0]
    assert "doc_id" in first
    assert "page_start" in first
    assert "page_end" in first
    assert first["page_start"] <= first["page_end"]
