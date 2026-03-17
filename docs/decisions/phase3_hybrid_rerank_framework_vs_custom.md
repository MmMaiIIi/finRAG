# Phase 3 Hybrid + Rerank Design: Framework Reuse vs Custom Logic

## Reused Framework Pieces

- LlamaIndex dense retrieval path is unchanged:
  - `VectorStoreIndex`
  - FAISS vector store integration
  - persisted storage and retriever API
- Existing LlamaIndex chunk/node artifacts from Phase 2 are reused directly.

## Thin Custom Logic Added

- Hybrid fusion:
  - explicit Reciprocal Rank Fusion (RRF) over dense and BM25 candidate lists
  - inspectable fusion contributions (`dense_rank`, `bm25_rank`, `rrf_score`)
- Rerank stage:
  - reranker adapter for `bge-reranker-v2-m3`
  - fallback adapter path kept explicit (`FlagEmbedding` -> cross-encoder)
  - optional score threshold policy
- Output normalization:
  - preserve `doc_id`, `title`, `source_path`, `page_start`, `page_end`
  - expose intermediate and final ranking scores for debugging

## Why This Fits AGENTS.md

- No custom retriever framework was introduced.
- LlamaIndex remains orchestration for indexing and dense retrieval.
- Custom code only bridges hybrid fusion visibility and rerank integration.
