# Phase 1 Parser Design: Framework Reuse vs Custom Logic

## Framework Reuse

- Keep downstream format aligned with LlamaIndex by exporting node-ready records (`text` + `metadata`) for direct `Document` creation in later phases.
- Use mature PDF backends from AGENTS requirements:
  - PyMuPDF as the primary extractor
  - pdfplumber as fallback/table-sensitive support

## Minimal Custom Logic

- Text cleanup tailored for inspectable Chinese financial report pages:
  - normalize whitespace/newlines
  - keep output human-readable
- Page-level provenance fields required for citations:
  - `doc_id`, `title`, `source_path`, `page_num`
- Parsing warnings per page and per document:
  - empty page warning
  - fallback usage warning
  - parser failure warning tags
- Artifact persistence under `data/parsed/`:
  - pretty JSON document artifact
  - JSONL page records
  - JSONL node-ready records

## Out of Scope in Phase 1

- OCR-heavy extraction
- Index construction
- Retrieval/reranking/generation logic
