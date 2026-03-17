from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_fin.schemas import RetrievalResult


class Reranker(Protocol):
    """Simple reranker interface."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relevance scores for (query, candidate_text) pairs."""


@dataclass
class MockReranker:
    """Deterministic lightweight reranker for tests/local smoke runs."""

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, text in pairs:
            q_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
            t_terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower()))
            overlap = float(len(q_terms & t_terms))
            length_bonus = min(len(text), 400) / 1000.0
            scores.append(overlap + length_bonus)
        return scores


@dataclass
class FlagEmbeddingReranker:
    """FlagEmbedding reranker adapter."""

    model_name: str
    use_fp16: bool = False

    def __post_init__(self) -> None:
        from FlagEmbedding import FlagReranker

        self._reranker = FlagReranker(self.model_name, use_fp16=self.use_fp16)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._reranker.compute_score(pairs)
        if isinstance(scores, float):
            return [scores]
        return [float(s) for s in scores]


@dataclass
class CrossEncoderReranker:
    """sentence-transformers cross-encoder adapter."""

    model_name: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._reranker = CrossEncoder(self.model_name, device=self.device)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._reranker.predict(pairs)
        return [float(s) for s in scores]


def build_reranker(
    *,
    model_name: str,
    backend: str = "auto",
    device: str = "cpu",
    use_fp16: bool = False,
) -> Reranker:
    """Build reranker backend with transparent fallback behavior."""
    normalized_name = model_name.strip().lower()
    normalized_backend = backend.strip().lower()

    if normalized_name.startswith("mock"):
        return MockReranker()

    if normalized_backend in {"flagembedding", "auto"}:
        try:
            return FlagEmbeddingReranker(model_name=model_name, use_fp16=use_fp16)
        except Exception:
            if normalized_backend == "flagembedding":
                raise

    return CrossEncoderReranker(model_name=model_name, device=device)


def rerank_candidates(
    *,
    query: str,
    candidates: list[RetrievalResult],
    reranker: Reranker,
    rerank_top_n: int,
    rerank_score_threshold: float | None = None,
) -> list[RetrievalResult]:
    """Rerank fused candidates while preserving metadata and exposing scores."""
    if not candidates:
        return []

    pairs = [(query, item.text) for item in candidates]
    scores = reranker.score_pairs(pairs)

    reranked: list[tuple[RetrievalResult, float]] = []
    for pre_rank, (item, score) in enumerate(zip(candidates, scores, strict=True), start=1):
        if rerank_score_threshold is not None and score < rerank_score_threshold:
            continue

        meta = dict(item.metadata.metadata)
        meta["pre_rerank_rank"] = pre_rank
        meta["rerank_score"] = score
        item.metadata.metadata = meta

        reranked.append(
            (
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    score=score,
                    retrieval_source="rerank",
                    metadata=item.metadata,
                ),
                score,
            )
        )

    reranked.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in reranked[:rerank_top_n]]
