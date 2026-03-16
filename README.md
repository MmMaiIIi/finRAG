# finRAG – Project Status Summary

## Overview

This repository currently contains the **initial scaffold of a financial research report RAG system**.
The present implementation focuses on establishing the **engineering structure, configuration system, data schemas, CLI pipeline entry points, and testing framework**.

Core RAG functionality such as document parsing, indexing, retrieval, and answer generation has **not yet been implemented**.

The project is therefore in a **scaffold stage**, where the architecture is defined but most algorithmic components remain to be built.

---

## Project Structure

The repository follows a standard Python ML project layout:

```
finRAG/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ configs/
│  ├─ model.yaml
│  ├─ retrieval.yaml
│  └─ eval.yaml
├─ src/
│  ├─ core/
│  │  └─ schemas.py
│  ├─ serving/
│  │  └─ schemas.py
│  └─ utils/
│     ├─ io.py
│     ├─ logger.py
│     ├─ text.py
│     └─ config.py
├─ scripts/
│  ├─ parse_pdf.py
│  ├─ build_chunks.py
│  ├─ build_faiss.py
│  ├─ build_bm25.py
│  └─ run_retrieval_eval.py
└─ tests/
   ├─ test_schemas.py
   ├─ test_io.py
   └─ test_config.py
```

### Key Components

**configs/**
Configuration files for model settings, retrieval strategies, and evaluation parameters.

**src/core/**
Defines core data schemas used across the pipeline (documents, chunks, queries, retrieval outputs).

**src/serving/**
Schemas for serving and API interfaces.

**src/utils/**
Utility modules including configuration loading, logging, text processing, and IO helpers.

**scripts/**
Command-line entry points for the main pipeline stages.

**tests/**
Basic unit tests ensuring configuration, schema, and utility modules load correctly.

---

## Pipeline Entry Points

The following CLI scripts represent the intended RAG pipeline stages:

| Script                  | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `parse_pdf.py`          | Parse financial report PDFs into raw documents |
| `build_chunks.py`       | Convert documents into retrieval chunks        |
| `build_faiss.py`        | Construct dense vector index using FAISS       |
| `build_bm25.py`         | Build sparse retrieval index (BM25)            |
| `run_retrieval_eval.py` | Evaluate retrieval performance                 |

These scripts currently contain **only scaffolding** and do not yet implement full processing logic.

---

## Current Implementation Status

### Implemented

* Project directory structure
* Configuration system (`configs/`)
* Core data schemas
* CLI pipeline entry points
* Utility modules (logging, IO, config loader)
* Basic unit tests
* Import and compilation validation

### Verified

The following checks were executed successfully:

* `pytest`
  **6 tests passed**

* `compileall`
  Successful compilation for `src`, `scripts`, and `tests`

* Import validation

```python
import src
import src.core.schemas
import scripts.parse_pdf
import scripts.build_chunks
import scripts.build_faiss
import scripts.build_bm25
import scripts.run_retrieval_eval
```

All modules import successfully.

---

## Not Implemented Yet

The following functional components are intentionally left for future development:

* PDF parsing and extraction pipeline
* Chunking strategy and text segmentation
* Embedding generation
* FAISS dense index construction
* BM25 sparse index construction
* Retrieval runtime logic
* Reranking
* Answer generation with LLM
* Retrieval evaluation metrics
* Benchmarking or quality claims

---

## Development Roadmap

The next development stages will progressively implement the full RAG pipeline:

1. **PDF Parsing**

   * Extract structured text from financial research reports

2. **Document Chunking**

   * Segment documents into retrieval-friendly units

3. **Dense Retrieval**

   * Generate embeddings
   * Build FAISS index

4. **Sparse Retrieval**

   * Build BM25 index

5. **Hybrid Retrieval**

   * Combine dense and sparse search

6. **Answer Generation**

   * Integrate LLM for final response generation

7. **Evaluation**

   * Implement retrieval metrics and benchmarks

---

## Current Status

The repository currently provides a **clean and validated engineering scaffold** for a financial-report RAG system.

The architecture, configuration structure, and pipeline interfaces are defined, enabling incremental implementation of the retrieval and generation components in subsequent stages.
