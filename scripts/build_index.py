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
        description="Phase 0 placeholder for index building."
    )
    parser.add_argument("--config", default="default", help="Retrieval config name.")
    parser.add_argument(
        "--parsed-dir",
        default="data/parsed",
        help="Input parsed artifact directory.",
    )
    parser.add_argument(
        "--index-dir",
        default="data/indexes",
        help="Output index artifact directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path(
        "retrieval", args.config, PROJECT_ROOT / "configs"
    )
    _ = load_stage_config("retrieval", args.config, PROJECT_ROOT / "configs")

    print(f"[Phase0] Loaded retrieval config: {config_path}")
    print(f"[Phase0] Parsed dir: {args.parsed_dir}")
    print(f"[Phase0] Index dir: {args.index_dir}")
    print("[Phase0] Index build pipeline is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
