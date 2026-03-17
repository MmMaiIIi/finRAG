"""Retrieval modules for Phase 2 baseline."""

from rag_fin.retrieval.bm25 import bm25_retrieve
from rag_fin.retrieval.dense import dense_retrieve
from rag_fin.retrieval.formatting import format_retrieval_results
from rag_fin.retrieval.fusion import FusionConfig, rrf_fuse
from rag_fin.retrieval.runner import run_retrieval, save_retrieval_output

__all__ = [
    "dense_retrieve",
    "bm25_retrieve",
    "FusionConfig",
    "rrf_fuse",
    "format_retrieval_results",
    "run_retrieval",
    "save_retrieval_output",
]
