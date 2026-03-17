# finRAG (Phase 2: Retrieval Baseline)

Framework-first Chinese financial PDF RAG project using LlamaIndex as the primary orchestration layer.

## Current Stage

Phase 2 is implemented:

- Page-level parsed artifacts from Phase 1 are converted into LlamaIndex documents/nodes.
- Dense retrieval baseline is built with LlamaIndex + FAISS.
- Lexical retrieval baseline is built with `rank_bm25`.
- Retrieval outputs are normalized and saved with page citation metadata.

Out of scope in this phase:

- reranking
- answer generation
- refusal logic

## Retrieval Baseline Pipeline

1. Read parsed page artifacts (`*.pages.jsonl`) from `data/parsed/`.
2. Build LlamaIndex `Document` objects with provenance metadata.
3. Chunk with LlamaIndex `SentenceSplitter` (configurable chunk size/overlap).
4. Build dense FAISS index via LlamaIndex `VectorStoreIndex`.
5. Build BM25 corpus artifact via `rank_bm25`.
6. Save inspectable artifacts:
   - `chunks.jsonl`
   - `index_manifest.json`
   - `dense/` persisted LlamaIndex+FAISS files
   - `bm25/bm25_corpus.json`
   - retrieval run outputs (`retrieval_outputs/*.json`)

## Configs

- `configs/retrieval/default.json`: BGE model defaults
- `configs/retrieval/mock_local.json`: mock embedding profile for local smoke tests

Config fields include:

- `embedding_model`
- `chunk_size`
- `chunk_overlap`
- `dense_top_k`
- `bm25_top_k`

## Scripts

Build index:

```bash
python scripts/build_index.py --config mock_local --parsed-dir data/parsed --index-dir data/indexes/retrieval_mock
```

Inspect chunks:

```bash
python scripts/inspect_chunks.py --index-dir data/indexes/retrieval_mock --limit 5
```

Run retrieval:

```bash
python scripts/run_retrieval.py "revenue growth" --config mock_local --index-dir data/indexes/retrieval_mock --mode both
```

Run tests:

```bash
pytest
```

## Framework Reuse vs Custom

Framework reuse (LlamaIndex):

- `Document` representation
- `SentenceSplitter` node/chunk creation
- `VectorStoreIndex` dense retrieval orchestration
- FAISS vector-store integration through `llama-index-vector-stores-faiss`

Minimal custom code:

- loading parsed page artifacts and preserving page provenance metadata
- BM25 bridge (`rank_bm25`) and tokenizer for mixed Chinese/alphanumeric text
- result normalization into inspectable citation-rich JSON output
