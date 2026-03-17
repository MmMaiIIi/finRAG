# Phase 2 Retrieval Design: Framework Reuse vs Custom Logic

## Reused from LlamaIndex

- `Document` and node metadata flow
- `SentenceSplitter` for chunk creation
- `VectorStoreIndex` for dense retrieval orchestration
- FAISS vector store integration and persistence (`llama-index-vector-stores-faiss`)

## Minimal Custom Logic

- Load parsed page JSONL artifacts and enforce page-level provenance on each chunk.
- Thin BM25 bridge using `rank_bm25` for lexical retrieval.
- Output normalization into a shared retrieval result format with citation fields (`doc_id`, `page_start`, `page_end`).
- Inspectable artifact persistence (`chunks.jsonl`, BM25 corpus JSON, retrieval output JSON).

## Out of Scope in Phase 2

- reranking
- answer synthesis
- refusal policy
