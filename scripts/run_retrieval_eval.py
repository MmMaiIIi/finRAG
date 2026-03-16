"""Phase-1 placeholder for retrieval evaluation entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_yaml_config
from src.utils.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against qrels.")
    parser.add_argument("--eval-config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--predictions-path", type=Path, default=Path("artifacts/eval/predictions.jsonl"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("scripts.run_retrieval_eval")
    _ = load_yaml_config(args.eval_config)
    logger.info("Phase 1 placeholder only. Retrieval evaluation is not implemented yet.")
    logger.info("eval_config=%s predictions_path=%s", args.eval_config, args.predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

