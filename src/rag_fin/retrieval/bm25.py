from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.schema import BaseNode
from rank_bm25 import BM25Okapi

from rag_fin.retrieval.normalize import chunk_record_to_result
from rag_fin.schemas import RetrievalResult


def tokenize_for_bm25(text: str) -> list[str]:
    """Lightweight tokenizer that works for Chinese chars and alphanumeric terms."""
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", lowered)
    return tokens if tokens else ["<empty>"]


def build_bm25_artifact(
    *,
    nodes: list[BaseNode],
    output_dir: str | Path,
) -> Path:
    """Save inspectable BM25 corpus artifact for later retrieval."""
    bm25_dir = Path(output_dir)
    bm25_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for node in nodes:
        text = node.get_content().strip()
        records.append(
            {
                "chunk_id": node.node_id,
                "text": text,
                "tokens": tokenize_for_bm25(text),
                "metadata": node.metadata,
            }
        )

    payload = {
        "tokenizer": "mixed_zh_en_char_word_v1",
        "chunks": records,
    }
    artifact_path = bm25_dir / "bm25_corpus.json"
    artifact_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_path


def load_bm25_artifact(index_dir: str | Path) -> dict[str, Any]:
    """Load BM25 artifact from index directory."""
    path = Path(index_dir) / "bm25" / "bm25_corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def bm25_retrieve(
    *,
    query: str,
    index_dir: str | Path,
    top_k: int,
) -> list[RetrievalResult]:
    """Run rank_bm25 retrieval and normalize output format."""
    artifact = load_bm25_artifact(index_dir)
    chunks: list[dict[str, Any]] = artifact["chunks"]
    tokenized_corpus = [chunk["tokens"] for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[RetrievalResult] = []
    for idx in ranked_indices:
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])
        results.append(
            chunk_record_to_result(
                chunk=chunk,
                score=score,
                retrieval_source="bm25",
            )
        )
    return results
