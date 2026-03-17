from __future__ import annotations

from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from rag_fin.retrieval.normalize import node_with_score_to_result
from rag_fin.schemas import RetrievalResult


def build_embedding_model(
    *,
    model_name: str,
    device: str = "cpu",
    mock_embedding_dim: int = 64,
) -> BaseEmbedding:
    """Build embedding model for dense indexing/retrieval."""
    if model_name.strip().lower().startswith("mock"):
        return MockEmbedding(embed_dim=mock_embedding_dim)
    return HuggingFaceEmbedding(model_name=model_name, device=device)


def load_dense_nodes(
    *,
    index_dir: str | Path,
    embedding_model: BaseEmbedding,
):
    """Load persisted LlamaIndex dense index from FAISS artifacts."""
    dense_dir = Path(index_dir) / "dense"
    vector_store = FaissVectorStore.from_persist_dir(str(dense_dir))
    storage_context = StorageContext.from_defaults(
        persist_dir=str(dense_dir),
        vector_store=vector_store,
    )
    return load_index_from_storage(storage_context, embed_model=embedding_model)


def dense_retrieve(
    *,
    query: str,
    index_dir: str | Path,
    embedding_model: BaseEmbedding,
    top_k: int,
) -> list[RetrievalResult]:
    """Run dense retrieval and normalize into shared retrieval result schema."""
    index = load_dense_nodes(index_dir=index_dir, embedding_model=embedding_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes: list[NodeWithScore] = retriever.retrieve(query)
    return [node_with_score_to_result(item, retrieval_source="dense") for item in nodes]
