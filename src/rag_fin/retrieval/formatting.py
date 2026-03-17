from __future__ import annotations

from typing import Any

from rag_fin.schemas import RetrievalResult


def citation_label(result: RetrievalResult) -> str:
    """Build compact page citation label for retrieval outputs."""
    page_start = result.metadata.page_start
    page_end = result.metadata.page_end
    if page_start == page_end:
        page_label = f"p.{page_start}"
    else:
        page_label = f"p.{page_start}-{page_end}"
    return f"{result.metadata.doc_id} {page_label}"


def format_retrieval_results(
    *,
    query: str,
    results: list[RetrievalResult],
    retriever_name: str,
) -> dict[str, Any]:
    """Normalize retrieval results into inspectable JSON-friendly payload."""
    items: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        debug: dict[str, Any] = {}
        meta = item.metadata.metadata
        for key in (
            "dense_rank",
            "dense_score",
            "bm25_rank",
            "bm25_score",
            "rrf_score",
            "pre_rerank_rank",
            "rerank_score",
        ):
            if key in meta:
                debug[key] = meta[key]

        items.append(
            {
                "rank": rank,
                "score": item.score,
                "retrieval_source": item.retrieval_source,
                "chunk_id": item.chunk_id,
                "doc_id": item.metadata.doc_id,
                "title": item.metadata.title,
                "source_path": item.metadata.source_path,
                "page_start": item.metadata.page_start,
                "page_end": item.metadata.page_end,
                "citation": citation_label(item),
                "text": item.text,
                "debug": debug,
            }
        )

    return {
        "query": query,
        "retriever": retriever_name,
        "count": len(items),
        "results": items,
    }
