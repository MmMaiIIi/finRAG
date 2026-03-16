"""Phase-1 placeholder for BM25 index build entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a BM25 index from chunk records.")
    parser.add_argument("--chunks-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-path", type=Path, default=Path("artifacts/index/bm25.pkl"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("configs/retrieval.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("scripts.build_bm25")
    _ = load_yaml_config(args.retrieval_config)
    logger.info("Phase 1 placeholder only. BM25 indexing is not implemented yet.")
    logger.info("chunks_path=%s index_path=%s", args.chunks_path, args.index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

