# AGENTS.md

This repository is a portfolio-grade Chinese financial PDF RAG system.

Global rules for Codex:
1. Do not redesign the whole repository unless explicitly asked.
2. Work in small, reviewable steps.
3. Keep code modular, typed, and explainable by an intern candidate.
4. Prefer local files, JSONL, YAML configs, and simple scripts over heavy infrastructure.
5. Do not fabricate benchmark results or claim success without runnable tests.
6. Every phase must end with:
   - files changed
   - commands run
   - test results
   - known limitations
7. Preserve source metadata carefully:
   - doc_id
   - file_name
   - page_no
   - chunk_id
   - chunk_type
8. Any answer generation must be grounded in retrieved evidence only.
9. If a requested feature depends on unavailable credentials or external APIs, implement a clean adapter and a local stub path.
10. Keep the repository runnable on a local machine.

Coding rules:
- Python 3.10+
- Type hints required for public functions
- Use dataclasses or pydantic where helpful, but avoid overengineering
- Add docstrings to non-trivial modules
- Keep dependencies minimal
- Use pytest for tests
- Use logging, not print, in library code

Testing rules:
- Add unit tests for pure logic
- Add at least one integration test per phase where practical
- Do not mark a phase complete if the required tests fail

Output rules:
- Be explicit about what is implemented
- Be explicit about what is still stubbed or approximate
- Never claim production-grade robustness unless it is actually validated