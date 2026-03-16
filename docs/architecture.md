# Architecture Notes (Phase 0)

This repository follows a framework-first design centered on LlamaIndex.

## Target Pipeline (Implemented in Later Phases)

1. Load PDF files
2. Parse pages/tables
3. Normalize text
4. Build chunks/nodes
5. Build FAISS dense index
6. Build BM25 retriever
7. Hybrid retrieval
8. Reranking
9. Answer synthesis with citations
10. Refusal for weak/contradictory evidence
11. Evaluation and badcase logging

Phase 0 only scaffolds structure, types, configs, and tests.
