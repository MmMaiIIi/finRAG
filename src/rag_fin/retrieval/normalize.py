from __future__ import annotations

from typing import Any

from llama_index.core.schema import NodeWithScore

from rag_fin.schemas import ChunkMetadata, RetrievalResult


def _chunk_metadata_from_mapping(
    *,
    chunk_id: str,
    mapping: dict[str, Any],
) -> ChunkMetadata:
    page_num = int(mapping.get("page_num", 1))
    page_start = int(mapping.get("page_start", page_num))
    page_end = int(mapping.get("page_end", page_start))
    doc_id = str(mapping.get("doc_id", "unknown_doc"))
    title = str(mapping.get("title", doc_id))
    source_path = str(mapping.get("source_path", ""))

    return ChunkMetadata(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=title,
        source_path=source_path,
        page_start=page_start,
        page_end=page_end,
        metadata=mapping,
    )


def node_with_score_to_result(
    item: NodeWithScore,
    *,
    retrieval_source: str,
) -> RetrievalResult:
    """Convert LlamaIndex retrieval item into shared output schema."""
    mapping = dict(item.node.metadata)
    chunk_meta = _chunk_metadata_from_mapping(chunk_id=item.node.node_id, mapping=mapping)
    return RetrievalResult(
        chunk_id=item.node.node_id,
        text=item.node.get_content(),
        score=float(item.score) if item.score is not None else 0.0,
        retrieval_source=retrieval_source,
        metadata=chunk_meta,
    )


def chunk_record_to_result(
    *,
    chunk: dict[str, Any],
    score: float,
    retrieval_source: str,
) -> RetrievalResult:
    """Convert stored chunk record into shared output schema."""
    chunk_id = str(chunk["chunk_id"])
    text = str(chunk["text"])
    mapping = dict(chunk["metadata"])
    chunk_meta = _chunk_metadata_from_mapping(chunk_id=chunk_id, mapping=mapping)
    return RetrievalResult(
        chunk_id=chunk_id,
        text=text,
        score=score,
        retrieval_source=retrieval_source,
        metadata=chunk_meta,
    )
