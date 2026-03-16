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
        description="Phase 0 placeholder for question answering."
    )
    parser.add_argument("question", help="User question.")
    parser.add_argument("--config", default="default", help="Generation config name.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path(
        "generation", args.config, PROJECT_ROOT / "configs"
    )
    _ = load_stage_config("generation", args.config, PROJECT_ROOT / "configs")

    print(f"[Phase0] Loaded generation config: {config_path}")
    print(f"[Phase0] Question: {args.question}")
    print("[Phase0] Query pipeline is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
