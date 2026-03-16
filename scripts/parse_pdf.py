"""Phase-1 placeholder for PDF parsing entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDF files into structured pages/tables.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-path", type=Path, default=Path("data/parsed/pages.jsonl"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("scripts.parse_pdf")
    _ = load_yaml_config(args.model_config)
    logger.info("Phase 1 placeholder only. Parsing is not implemented yet.")
    logger.info("input_dir=%s output_path=%s", args.input_dir, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

