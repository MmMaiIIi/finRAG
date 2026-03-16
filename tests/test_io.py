"""Tests for JSONL IO helper utilities."""

from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl


def test_jsonl_read_write_round_trip(tmp_path: Path) -> None:
    records = [
        {"id": "1", "text": "first"},
        {"id": "2", "text": "second"},
    ]
    output_path = tmp_path / "sample.jsonl"

    written = write_jsonl(output_path, records)
    loaded = list(read_jsonl(output_path))

    assert written == 2
    assert loaded == records


def test_read_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    output_path = tmp_path / "with_blank_lines.jsonl"
    output_path.write_text('{"id":"1"}\n\n{"id":"2"}\n', encoding="utf-8")

    loaded = list(read_jsonl(output_path))

    assert loaded == [{"id": "1"}, {"id": "2"}]
