# finRAG (Phase 3: Hybrid Retrieval + Reranking)

Framework-first Chinese financial PDF RAG project using LlamaIndex as the main orchestration layer.

## Current Stage

Phase 3 is implemented:

- Dense retrieval: LlamaIndex + FAISS
- Lexical retrieval: `rank_bm25`
- Hybrid fusion: explicit RRF
- Reranking: `bge-reranker-v2-m3` adapter path
- Inspectable intermediate scores and final ranking outputs

Out of scope in this phase:

- generation/refusal policies
- full evaluation loop

## Pipeline

1. Parse pages (Phase 1 artifacts)
2. Build nodes/chunks + dense index + BM25 corpus (Phase 2 baseline)
3. Query-time retrieval:
   - dense top-k
   - BM25 top-k
   - RRF fuse candidates
   - rerank fused candidates
   - return final top evidence with page metadata

## Scripts

Build baseline index artifacts:

```bash
python scripts/build_index.py --config mock_local --parsed-dir data/parsed --index-dir data/indexes/retrieval_mock
```

Inspect chunk artifacts:

```bash
python scripts/inspect_chunks.py --index-dir data/indexes/retrieval_mock --limit 5
```

Run hybrid retrieval + rerank:

```bash
python scripts/run_retrieval.py "manufacturing support" --config mock_local --index-dir data/indexes/retrieval_mock --mode hybrid
```

## Retrieval Config Highlights

See `configs/retrieval/default.json` and `configs/retrieval/mock_local.json`:

- `dense_top_k`
- `bm25_top_k`
- `fused_top_n`
- `rerank_top_n`
- `fusion_strategy` (`rrf`)
- `rrf_k`
- `reranker_model`
- `rerank_score_threshold`

## Inspectable Outputs

Under index directory:

- `chunks.jsonl`
- `index_manifest.json`
- `dense/`
- `bm25/bm25_corpus.json`
- `retrieval_outputs/*.json` with:
  - dense ranking
  - BM25 ranking
  - fused ranking
  - reranked final ranking
  - score/debug fields and page citation metadata

## Framework Reuse vs Custom

Reused from LlamaIndex:

- `Document` + `SentenceSplitter`
- `VectorStoreIndex` + FAISS vector store
- dense retriever loading/persistence

Thin custom additions:

- BM25 bridge (`rank_bm25`)
- RRF fusion layer
- reranker adapter and scoring visibility
- output normalization with citation-ready metadata
