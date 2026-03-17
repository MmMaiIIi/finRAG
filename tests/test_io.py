import subprocess
import sys
from pathlib import Path

from rag_fin.utils.config import load_stage_config
from rag_fin.utils.io import read_jsonl


def test_load_stage_config_parser_default() -> None:
    cfg = load_stage_config("parser", "default")
    assert cfg["pdf_backend_primary"] == "pymupdf"
    assert cfg["preserve_page_level_provenance"] is True


def test_read_synthetic_page_jsonl() -> None:
    records = read_jsonl("data/demo_samples/synthetic_pages.jsonl")
    assert len(records) >= 2
    assert records[0]["doc_id"] == "doc_fin_001"
    assert records[0]["page_num"] == 1


def test_cli_placeholders_show_help() -> None:
    scripts = [
        "ingest_and_parse.py",
        "build_index.py",
        "inspect_chunks.py",
        "run_retrieval.py",
        "ask.py",
        "run_eval.py",
        "launch_demo.py",
    ]
    for script_name in scripts:
        script_path = Path("scripts") / script_name
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
