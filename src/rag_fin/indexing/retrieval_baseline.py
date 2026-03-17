from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_fin.retrieval.bm25 import build_bm25_artifact
from rag_fin.retrieval.dense import build_embedding_model
from rag_fin.schemas import ParsedPageRecord
from rag_fin.utils.io import read_jsonl, write_jsonl


class RetrievalBuildConfig(BaseModel):
    """Config for Phase 2 retrieval baseline build."""

    model_config = ConfigDict(extra="forbid")

    dense_backend: str = "faiss"
    sparse_backend: str = "rank_bm25"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"
    mock_embedding_dim: int = Field(default=64, ge=8)
    chunk_size: int = Field(default=300, ge=20)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k: int = Field(default=5, ge=1)
    dense_top_k: int | None = Field(default=None, ge=1)
    bm25_top_k: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_chunking(self) -> "RetrievalBuildConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def resolved_dense_top_k(self) -> int:
        return self.dense_top_k if self.dense_top_k is not None else self.top_k

    @property
    def resolved_bm25_top_k(self) -> int:
        return self.bm25_top_k if self.bm25_top_k is not None else self.top_k


def load_parsed_pages(parsed_dir: str | Path) -> list[ParsedPageRecord]:
    """Load parsed page records from *.pages.jsonl artifacts."""
    directory = Path(parsed_dir)
    page_files = sorted(directory.glob("*.pages.jsonl"))
    pages: list[ParsedPageRecord] = []

    for page_file in page_files:
        for row in read_jsonl(page_file):
            pages.append(ParsedPageRecord.model_validate(row))

    if not pages:
        raise FileNotFoundError(f"no parsed page artifacts found in {directory.as_posix()}")
    return pages


def build_documents_from_pages(
    pages: list[ParsedPageRecord],
) -> tuple[list[Document], int]:
    """Convert parsed pages into LlamaIndex Documents for chunking."""
    documents: list[Document] = []
    skipped_empty = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            skipped_empty += 1
            continue

        metadata = {
            "doc_id": page.doc_id,
            "title": page.title,
            "source_path": page.source_path,
            "page_num": page.page_num,
            "page_start": page.page_num,
            "page_end": page.page_num,
            "parsing_warnings": page.parsing_warnings,
            **page.metadata,
        }
        documents.append(
            Document(
                text=text,
                id_=f"{page.doc_id}:p{page.page_num}",
                metadata=metadata,
            )
        )

    return documents, skipped_empty


def build_nodes(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[BaseNode]:
    """Build chunked nodes with metadata preserved on each node."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    for node in nodes:
        meta = dict(node.metadata)
        page_num = int(meta.get("page_num", 1))
        meta["page_start"] = int(meta.get("page_start", page_num))
        meta["page_end"] = int(meta.get("page_end", page_num))
        meta["doc_id"] = str(meta.get("doc_id", "unknown_doc"))
        meta["title"] = str(meta.get("title", ""))
        meta["source_path"] = str(meta.get("source_path", ""))
        node.metadata = meta
    return nodes


def save_node_artifacts(nodes: list[BaseNode], output_path: str | Path) -> Path:
    """Save inspectable chunk artifacts with text + metadata."""
    path = Path(output_path)
    records: list[dict[str, Any]] = []
    for node in nodes:
        records.append(
            {
                "chunk_id": node.node_id,
                "text": node.get_content(),
                "metadata": node.metadata,
            }
        )
    write_jsonl(path, records)
    return path


def build_dense_index(
    nodes: list[BaseNode],
    config: RetrievalBuildConfig,
    dense_dir: str | Path,
) -> Path:
    """Build and persist FAISS-backed dense index via LlamaIndex."""
    dense_path = Path(dense_dir)
    dense_path.mkdir(parents=True, exist_ok=True)

    embedding_model = build_embedding_model(
        model_name=config.embedding_model,
        device=config.embedding_device,
        mock_embedding_dim=config.mock_embedding_dim,
    )
    embedding_dim = len(embedding_model.get_text_embedding("embedding dimension probe"))

    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(embedding_dim))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=False,
    )
    index.storage_context.persist(persist_dir=str(dense_path))
    return dense_path


def build_retrieval_baseline(
    *,
    parsed_dir: str | Path,
    index_dir: str | Path,
    config: RetrievalBuildConfig,
) -> dict[str, Any]:
    """End-to-end Phase 2 build: chunking + dense index + BM25 artifacts."""
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = load_parsed_pages(parsed_dir)
    documents, skipped_empty_pages = build_documents_from_pages(pages)
    nodes = build_nodes(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    chunks_path = save_node_artifacts(nodes=nodes, output_path=out_dir / "chunks.jsonl")
    dense_path = build_dense_index(nodes=nodes, config=config, dense_dir=out_dir / "dense")
    bm25_path = build_bm25_artifact(nodes=nodes, output_dir=out_dir / "bm25")

    manifest = {
        "stage": "phase2_retrieval_baseline",
        "dense_backend": config.dense_backend,
        "sparse_backend": config.sparse_backend,
        "embedding_model": config.embedding_model,
        "embedding_device": config.embedding_device,
        "mock_embedding_dim": config.mock_embedding_dim,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
        "dense_top_k": config.resolved_dense_top_k,
        "bm25_top_k": config.resolved_bm25_top_k,
        "parsed_pages": len(pages),
        "skipped_empty_pages": skipped_empty_pages,
        "documents_indexed": len(documents),
        "chunks_indexed": len(nodes),
        "artifacts": {
            "chunks_jsonl": str(chunks_path.as_posix()),
            "dense_dir": str(dense_path.as_posix()),
            "bm25_json": str(bm25_path.as_posix()),
        },
    }

    manifest_path = out_dir / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_index_manifest(index_dir: str | Path) -> dict[str, Any]:
    """Load index manifest from an index directory."""
    manifest_path = Path(index_dir) / "index_manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
