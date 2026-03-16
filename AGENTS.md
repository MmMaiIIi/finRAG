# AGENTS.md

## Project
Financial Report RAG System based on Existing Frameworks

This repository builds a Chinese RAG system for financial research reports and policy PDFs.
The project must prioritize mature frameworks and reusable components over rebuilding core infrastructure from scratch.

The system should answer user questions with grounded evidence, document/page citations, and safe refusal when evidence is insufficient.

## Primary Development Principle
Framework-first. Minimal custom code.

Do not rebuild standard RAG components if mature open-source solutions already exist.
Custom code should only be written when it creates clear value for this project.

## Preferred Stack

### Primary Framework
- LlamaIndex

Use LlamaIndex as the main orchestration layer for:
- document ingestion abstraction
- node / chunk pipeline
- retriever composition
- query engine wiring
- response synthesis integration

### Retrieval
- FAISS for local dense vector retrieval
- rank_bm25 for lexical retrieval

### Embedding / Reranking
- bge-large-zh-v1.5 for embeddings
- bge-reranker-v2-m3 for reranking

### LLM
- Qwen3-8B or Qwen2.5-7B

### PDF Parsing
- PyMuPDF for primary parsing
- pdfplumber for table-sensitive extraction support

### Evaluation
- RAGAS
- custom offline financial QA evaluation set

## Reference Repositories
These repositories are references, not primary implementation targets.

### RAGFlow
Study for:
- production RAG architecture
- document parsing workflow ideas
- evaluation and badcase loop

### Langchain-Chatchat
Study for:
- Chinese RAG engineering patterns
- hybrid retrieval ideas
- knowledge base workflow design

### QAnything
Study for:
- two-stage retrieval
- parent-child retrieval
- rerank threshold strategy

### LlamaIndex
This is the primary framework for implementation.

### FAISS
Use as the default dense retrieval backend.

### FlagEmbedding
Use as the default embedding and reranking toolkit.

### Milvus
Optional later-stage backend for larger-scale corpus support.

### GraphRAG
Out of scope for MVP. Only optional future exploration.

## Product Goal
Given a Chinese user query, retrieve relevant evidence from financial-report and policy PDFs, then generate a grounded answer with citations.

## MVP Requirements
1. Ingest and parse a corpus of Chinese PDFs.
2. Preserve document metadata and page numbers.
3. Build dense retrieval and BM25 retrieval.
4. Support hybrid retrieval.
5. Support reranking.
6. Generate answers with citations.
7. Refuse when evidence is insufficient.
8. Provide offline evaluation and badcase logging.
9. Expose a simple demo UI.

## Non-Goals
- Do not build a custom RAG framework from scratch.
- Do not build a custom vector database.
- Do not rebuild document stores already handled well by LlamaIndex.
- Do not introduce GraphRAG in MVP.
- Do not add excessive abstraction layers with no measurable benefit.

## Custom Code Allowed
Custom code is expected only in the following areas:
1. financial PDF parsing cleanup
2. Chinese chunking and metadata preservation
3. hybrid retrieval fusion if needed beyond framework defaults
4. reranker integration and threshold policy
5. citation formatting and validation
6. refusal logic
7. evaluation pipeline and badcase analysis
8. demo observability

## Architecture Direction
The default pipeline should be:

1. Load PDFs
2. Parse pages and tables
3. Clean and normalize text
4. Build nodes / chunks
5. Create dense index with FAISS
6. Build BM25 retriever
7. Combine retrievers into a hybrid retrieval pipeline
8. Rerank retrieved candidates
9. Generate answer from top evidence
10. Attach document and page citations
11. Refuse when confidence is weak
12. Log outputs for evaluation

## Framework Usage Rules
1. Prefer LlamaIndex built-in components whenever they meet the need.
2. If a built-in component is insufficient, extend it with small custom adapters instead of replacing the whole pipeline.
3. Keep framework usage explicit and inspectable.
4. Avoid hidden framework magic that makes debugging hard.
5. Every important stage must expose intermediate artifacts.

## Data Requirements
The corpus domain is:
- Chinese financial research reports
- policy PDFs
- optionally company announcements

Every parsed or indexed unit must preserve:
- doc_id
- title
- source file path
- page number range
- chunk text
- metadata

## Citation Requirement
Every final answer must include source citations.
At minimum each citation must identify:
- document title or doc_id
- page number or page range

Never fabricate citations.

## Refusal Requirement
The system must refuse to answer when:
- retrieval evidence is weak
- evidence is contradictory
- relevant context is missing
- the answer would require unsupported inference

## Evaluation Requirement
This project is incomplete without evaluation.

Track at minimum:
- Recall@5 / Recall@10 for retrieval
- citation validity
- answer correctness on a dev subset
- refusal precision on unanswerable queries
- latency per major stage

## Repository Structure
.
├─ AGENTS.md
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ parser/
│  ├─ retrieval/
│  ├─ generation/
│  └─ eval/
├─ data/
│  ├─ raw_pdfs/
│  ├─ parsed/
│  ├─ indexes/
│  ├─ eval/
│  └─ demo_samples/
├─ docs/
│  ├─ architecture.md
│  ├─ experiments/
│  ├─ decisions/
│  └─ badcases/
├─ scripts/
│  ├─ ingest_and_parse.py
│  ├─ build_index.py
│  ├─ run_eval.py
│  ├─ ask.py
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
│     └─ utils/
└─ tests/

## Stage Discipline
Implement only the requested stage.
Do not prematurely add advanced features.
Keep every stage reviewable and testable.

## Coding Rules
1. Use typed Python.
2. Prefer direct and simple code.
3. Keep framework wiring readable.
4. Avoid unnecessary wrappers.
5. Use config files for model names and thresholds.
6. Save intermediate artifacts for debugging.
7. Add tests for every nontrivial custom module.

## Deliverable Standard
The final repository should show:
- mature framework usage
- minimal but meaningful custom engineering
- measurable retrieval quality improvements
- grounded answering with citations
- demo and evaluation artifacts
- strong interview narrative

## Definition of Done
The project is done when a reviewer can:
- parse PDFs
- build the knowledge base
- ask Chinese questions
- inspect retrieved evidence
- verify citations
- see refusal behavior
- run evaluation
- understand what was reused vs customized