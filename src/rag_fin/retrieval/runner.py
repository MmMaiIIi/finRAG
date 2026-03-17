from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_fin.retrieval.bm25 import bm25_retrieve
from rag_fin.retrieval.dense import build_embedding_model, dense_retrieve
from rag_fin.retrieval.formatting import format_retrieval_results
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
) -> dict[str, Any]:
    """Run dense and/or BM25 retrieval and return formatted output payload."""
    dense_results: list[RetrievalResult] = []
    bm25_results: list[RetrievalResult] = []

    if mode in {"dense", "both"}:
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

    if mode in {"bm25", "both"}:
        bm25_results = bm25_retrieve(
            query=query,
            index_dir=index_dir,
            top_k=bm25_top_k,
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
