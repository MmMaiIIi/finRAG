"""Phase-1 placeholder for chunk building entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build retrievable chunks from parsed documents.")
    parser.add_argument("--input-path", type=Path, default=Path("data/parsed/pages.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("configs/retrieval.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("scripts.build_chunks")
    _ = load_yaml_config(args.retrieval_config)
    logger.info("Phase 1 placeholder only. Chunk building is not implemented yet.")
    logger.info("input_path=%s output_path=%s", args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

