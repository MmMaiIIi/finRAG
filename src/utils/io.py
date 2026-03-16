"""Helpers for JSONL file IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    ensure_ascii: bool = False,
) -> int:
    """Write records to a JSONL file and return number of written rows."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dict(record), ensure_ascii=ensure_ascii))
            f.write("\n")
            row_count += 1
    return row_count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a JSONL file."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {input_path}") from exc

