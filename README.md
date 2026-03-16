# finRAG (Phase 0 Scaffold)

Framework-first Chinese financial PDF RAG project scaffold.  
This phase only prepares repository structure, dependencies, typed interfaces, configs, and smoke tests.

## Principles

- LlamaIndex is the primary orchestration layer.
- Reuse mature components first, customize only project-specific gaps.
- Preserve page-level provenance (`doc_id`, title, source path, page range).
- Keep intermediate artifacts inspectable.
- No fabricated citations, metrics, or claims.

## What Phase 0 Includes

- Repository layout from `AGENTS.md`
- Python project setup (`pyproject.toml`, `requirements.txt`)
- Config folders and default JSON configs
- Thin typed interfaces:
  - `ParsedPageRecord`
  - `ChunkMetadata`
  - `RetrievalResult`
  - `EvaluationSample`
- Config and JSONL utility functions
- CLI placeholders in `scripts/`:
  - `ingest_and_parse.py`
  - `build_index.py`
  - `ask.py`
  - `run_eval.py`
  - `launch_demo.py`
- Synthetic sample data and smoke tests

## What Phase 0 Explicitly Does Not Include

- Real PDF parsing pipeline
- Real dense/sparse retrieval
- Real reranking
- Real answer generation
- Real RAG evaluation loop

## Planned Framework Usage (Later Phases)

- Ingestion and node pipeline: LlamaIndex document + node abstractions
- Dense retrieval: LlamaIndex + FAISS backend
- Sparse retrieval: BM25 (rank_bm25), fused with dense results
- Reranking: FlagEmbedding reranker integration
- Query engine and response synthesis: LlamaIndex query engine
- Evaluation: RAGAS + project-specific offline QA set

## Repository Layout

```text
.
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ parser/default.json
│  ├─ retrieval/default.json
│  ├─ generation/default.json
│  └─ eval/default.json
├─ data/
│  ├─ raw_pdfs/
│  ├─ parsed/
│  ├─ indexes/
│  ├─ eval/synthetic_eval.jsonl
│  └─ demo_samples/synthetic_pages.jsonl
├─ docs/
│  ├─ architecture.md
│  ├─ experiments/
│  ├─ decisions/
│  └─ badcases/
├─ scripts/
│  ├─ ingest_and_parse.py
│  ├─ build_index.py
│  ├─ ask.py
│  ├─ run_eval.py
│  └─ launch_demo.py
├─ src/
│  └─ rag_fin/
│     ├─ loaders/
│     ├─ parsing/
│     ├─ indexing/
│     ├─ retrieval/
│     ├─ rerank/
│     ├─ generation/
│     ├─ eval/
│     ├─ demo/
│     ├─ utils/
│     └─ schemas.py
└─ tests/
```

## Quick Validation

```bash
pytest
```
