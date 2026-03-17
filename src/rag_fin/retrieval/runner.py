from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_fin.retrieval.bm25 import bm25_retrieve
from rag_fin.retrieval.dense import build_embedding_model, dense_retrieve
from rag_fin.retrieval.formatting import format_retrieval_results
from rag_fin.retrieval.fusion import FusionConfig, rrf_fuse
from rag_fin.rerank import build_reranker, rerank_candidates
from rag_fin.schemas import RetrievalResult


def run_retrieval(
    *,
    query: str,
    index_dir: str | Path,
    mode: str,
    embedding_model_name: str,
    embedding_device: str,
    mock_embedding_dim: int,
    dense_top_k: int,
    bm25_top_k: int,
    fused_top_n: int,
    rerank_top_n: int,
    fusion_strategy: str,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    reranker_model: str,
    reranker_backend: str,
    reranker_use_fp16: bool,
    rerank_score_threshold: float | None = None,
    fusion_score_threshold: float | None = None,
) -> dict[str, Any]:
    """Run dense/BM25/hybrid retrieval and return inspectable payload."""
    dense_results: list[RetrievalResult] = []
    bm25_results: list[RetrievalResult] = []
    fused_results: list[RetrievalResult] = []
    reranked_results: list[RetrievalResult] = []

    if mode in {"dense", "both", "hybrid"}:
        embedding_model = build_embedding_model(
            model_name=embedding_model_name,
            device=embedding_device,
            mock_embedding_dim=mock_embedding_dim,
        )
        dense_results = dense_retrieve(
            query=query,
            index_dir=index_dir,
            embedding_model=embedding_model,
            top_k=dense_top_k,
        )

    if mode in {"bm25", "both", "hybrid"}:
        bm25_results = bm25_retrieve(
            query=query,
            index_dir=index_dir,
            top_k=bm25_top_k,
        )

    if mode in {"both", "hybrid"}:
        fusion_cfg = FusionConfig(
            strategy=fusion_strategy,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            fused_top_n=fused_top_n,
            fusion_score_threshold=fusion_score_threshold,
        )
        fused_results = rrf_fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            config=fusion_cfg,
        )

        reranker = build_reranker(
            model_name=reranker_model,
            backend=reranker_backend,
            device=embedding_device,
            use_fp16=reranker_use_fp16,
        )
        reranked_results = rerank_candidates(
            query=query,
            candidates=fused_results,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
            rerank_score_threshold=rerank_score_threshold,
        )

    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "dense": format_retrieval_results(
            query=query,
            results=dense_results,
            retriever_name="dense_faiss",
        ),
        "bm25": format_retrieval_results(
            query=query,
            results=bm25_results,
            retriever_name="bm25_rank_bm25",
        ),
        "fused": format_retrieval_results(
            query=query,
            results=fused_results,
            retriever_name=f"hybrid_{fusion_strategy}",
        ),
        "reranked": format_retrieval_results(
            query=query,
            results=reranked_results,
            retriever_name=reranker_model,
        ),
        "config": {
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "fused_top_n": fused_top_n,
            "rerank_top_n": rerank_top_n,
            "fusion_strategy": fusion_strategy,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
            "reranker_model": reranker_model,
            "reranker_backend": reranker_backend,
            "reranker_use_fp16": reranker_use_fp16,
            "fusion_score_threshold": fusion_score_threshold,
            "rerank_score_threshold": rerank_score_threshold,
        },
    }
    return payload


def save_retrieval_output(
    *,
    payload: dict[str, Any],
    index_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Save retrieval payload to inspectable JSON file."""
    if output_path is not None:
        path = Path(output_path)
    else:
        out_dir = Path(index_dir) / "retrieval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"retrieve_{timestamp}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
