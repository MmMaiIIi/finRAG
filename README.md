# finRAG (Phase 1: Ingestion + Parsing)

Framework-first Chinese financial PDF RAG project based on LlamaIndex.

## Current Stage

Phase 1 is implemented: PDF ingestion and page-level parsing.

- Primary parser: PyMuPDF
- Optional fallback/table-sensitive support: pdfplumber
- Output artifacts are inspectable under `data/parsed/`
- Page-level provenance is preserved for future citations/retrieval

Phase 1 does **not** implement indexing, retrieval, reranking, or generation.

## Framework-First Architecture

LlamaIndex remains the primary orchestration layer for later stages:

1. Parse/normalize pages (current phase)
2. Build nodes/chunks (next)
3. Dense retrieval with FAISS
4. BM25 retrieval and hybrid fusion
5. Reranking
6. Grounded answer synthesis with citations and refusal

In Phase 1, parsed records are exported as node-ready JSONL (`text` + `metadata`) so they can map directly into LlamaIndex `Document` objects later.

## Parsing Output

Each parsed page includes:

- `doc_id`
- `title`
- `source_path`
- `page_num`
- `text` (cleaned)
- `metadata`
- `parsing_warnings`

Each parsed document artifact includes:

- document-level metadata
- page count
- list of parsed pages

## Key Commands

```bash
python scripts/ingest_and_parse.py --config default --write-bundle
pytest
```

## Directory Highlights

```text
configs/parser/
  default.json
  table_sensitive.json

data/raw_pdfs/
  *.pdf

data/parsed/
  <doc_id>.parsed.json
  <doc_id>.pages.jsonl
  <doc_id>.llamaindex.jsonl

src/rag_fin/parsing/pdf_parser.py
scripts/ingest_and_parse.py
tests/test_parsing.py
docs/decisions/phase1_parser_framework_vs_custom.md
```
