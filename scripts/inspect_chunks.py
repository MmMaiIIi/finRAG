from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_fin.utils.io import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect stored chunk/node artifacts.")
    parser.add_argument(
        "--index-dir",
        default="data/indexes",
        help="Index directory that contains chunks.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks to print.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    index_dir = (PROJECT_ROOT / args.index_dir).resolve()
    chunks_path = index_dir / "chunks.jsonl"

    records = read_jsonl(chunks_path)
    print(f"[Phase2] chunks file: {chunks_path.as_posix()}")
    print(f"[Phase2] total chunks: {len(records)}")

    for i, record in enumerate(records[: args.limit], start=1):
        meta = record.get("metadata", {})
        page_start = meta.get("page_start", meta.get("page_num"))
        page_end = meta.get("page_end", page_start)
        print(
            f"{i}. chunk_id={record.get('chunk_id')} "
            f"citation={meta.get('doc_id')} p.{page_start}-{page_end}"
        )
        print(f"   title={meta.get('title')}")
        print(f"   text={str(record.get('text', ''))[:120]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
