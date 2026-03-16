# AGENTS.md

## Project
financial-report-rag

## Mission
Build a portfolio-grade Chinese RAG system for financial reports and policy PDFs.

This repository is not a toy demo. It should demonstrate real engineering ability across:
- PDF parsing
- chunking
- indexing
- hybrid retrieval
- reranking
- grounded answer generation
- citation validation
- refusal logic
- offline evaluation
- demo serving

The final system should be explainable in an internship interview end to end.

---

## Product Goal
Given a user query in Chinese, retrieve relevant evidence from a local corpus of Chinese PDF documents and generate an answer that:
1. is grounded in retrieved evidence,
2. includes source citations and page numbers,
3. refuses to answer when evidence is insufficient.

Example queries:
- 宁德时代 2024 年动力电池装机量是多少？
- 对比 2023 和 2024 年光伏行业景气度变化
- 某政策对新能源车行业的影响是什么？

---

## Non-Goals
Do not do the following unless explicitly requested in a later phase:
- no large distributed infrastructure
- no production database
- no Kubernetes / Docker orchestration unless requested
- no heavy frontend app beyond a simple Gradio demo
- no over-engineered framework abstraction
- no fake benchmark numbers
- no hidden network dependencies that make local runs brittle

---

## Engineering Principles

### 1. Local-first
Everything should run locally with Python and local files.
Use JSONL, CSV, YAML, and FAISS index files before introducing external systems.

### 2. Modular but practical
Prefer clean modules over deep abstraction hierarchies.
A small intern project should still be easy to explain line by line.

### 3. Evidence before generation
Retrieval quality is the foundation.
Do not overfocus on LLM generation before the retrieval stack is measurable.

### 4. Measurable progress
Every phase must produce:
- working code
- a testable artifact
- explicit metrics
- acceptance criteria

### 5. No fabricated results
If a metric was not measured, mark it as TODO.
Never invent numbers for README, reports, or sample outputs.

### 6. Strong metadata discipline
Every chunk and answer must preserve provenance.
At minimum keep:
- doc_id
- file_name
- page_no
- chunk_id
- chunk_type

### 7. Prefer explicit configuration
Use YAML config files and `.env`.
Avoid hardcoding paths, model names, API keys, or magic constants.

---

## Repository Shape

Expected top-level structure:

```text
financial-report-rag/
  README.md
  AGENTS.md
  requirements.txt
  .env.example
  configs/
    model.yaml
    retrieval.yaml
    eval.yaml
  data/
    raw_pdfs/
    parsed/
    chunks/
    indices/
    eval/
  scripts/
    ingest_pdfs.py
    parse_pdf.py
    build_chunks.py
    build_faiss.py
    build_bm25.py
    run_retrieval_eval.py
    run_rerank_eval.py
    run_generation_eval.py
    build_badcase_pool.py
  src/
    parser/
    chunking/
    retrieval/
    generation/
    evaluation/
    serving/
    utils/
  app/
    gradio_app.py
  outputs/
    reports/
    figures/
    logs/
  tests/