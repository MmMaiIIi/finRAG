from __future__ import annotations

from dataclasses import dataclass

from rag_fin.schemas import RetrievalResult


@dataclass
class FusionConfig:
    """Fusion configuration for hybrid retrieval."""

    strategy: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    fused_top_n: int = 20
    fusion_score_threshold: float | None = None


def _clone_with_fusion_details(
    *,
    base: RetrievalResult,
    fusion_score: float,
    dense_rank: int | None,
    dense_score: float | None,
    bm25_rank: int | None,
    bm25_score: float | None,
) -> RetrievalResult:
    meta = dict(base.metadata.metadata)
    meta.update(
        {
            "dense_rank": dense_rank,
            "dense_score": dense_score,
            "bm25_rank": bm25_rank,
            "bm25_score": bm25_score,
            "rrf_score": fusion_score,
        }
    )
    base.metadata.metadata = meta
    return RetrievalResult(
        chunk_id=base.chunk_id,
        text=base.text,
        score=fusion_score,
        retrieval_source="hybrid_rrf",
        metadata=base.metadata,
    )


def rrf_fuse(
    *,
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    config: FusionConfig,
) -> list[RetrievalResult]:
    """Fuse dense and BM25 candidates with Reciprocal Rank Fusion."""
    if config.strategy.lower() != "rrf":
        raise ValueError("only 'rrf' fusion is supported in Phase 3")

    dense_by_id = {item.chunk_id: item for item in dense_results}
    bm25_by_id = {item.chunk_id: item for item in bm25_results}
    all_chunk_ids = set(dense_by_id) | set(bm25_by_id)

    fused: list[RetrievalResult] = []
    for chunk_id in all_chunk_ids:
        dense_item = dense_by_id.get(chunk_id)
        bm25_item = bm25_by_id.get(chunk_id)

        dense_rank = (
            next((idx for idx, item in enumerate(dense_results, start=1) if item.chunk_id == chunk_id), None)
        )
        bm25_rank = (
            next((idx for idx, item in enumerate(bm25_results, start=1) if item.chunk_id == chunk_id), None)
        )

        score = 0.0
        if dense_rank is not None:
            score += config.dense_weight / (config.rrf_k + dense_rank)
        if bm25_rank is not None:
            score += config.bm25_weight / (config.rrf_k + bm25_rank)

        if config.fusion_score_threshold is not None and score < config.fusion_score_threshold:
            continue

        base = dense_item if dense_item is not None else bm25_item
        if base is None:  # pragma: no cover
            continue

        fused.append(
            _clone_with_fusion_details(
                base=base,
                fusion_score=score,
                dense_rank=dense_rank,
                dense_score=(dense_item.score if dense_item is not None else None),
                bm25_rank=bm25_rank,
                bm25_score=(bm25_item.score if bm25_item is not None else None),
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    return fused[: config.fused_top_n]
