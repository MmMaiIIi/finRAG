from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_fin.utils.config import load_stage_config, resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 placeholder for ingestion and parsing."
    )
    parser.add_argument("--config", default="default", help="Parser config name.")
    parser.add_argument(
        "--input-dir",
        default="data/raw_pdfs",
        help="Input PDF directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/parsed",
        help="Output parsed artifact directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path("parser", args.config, PROJECT_ROOT / "configs")
    _ = load_stage_config("parser", args.config, PROJECT_ROOT / "configs")

    print(f"[Phase0] Loaded parser config: {config_path}")
    print(f"[Phase0] Input dir: {args.input_dir}")
    print(f"[Phase0] Output dir: {args.output_dir}")
    print("[Phase0] Ingest/parse pipeline is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
