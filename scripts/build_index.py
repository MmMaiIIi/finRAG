from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_fin.indexing import RetrievalBuildConfig, build_retrieval_baseline
from rag_fin.utils.config import load_stage_config, resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3 index build for hybrid retrieval.")
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
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override (e.g., BAAI/bge-large-zh-v1.5 or mock).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path(
        "retrieval", args.config, PROJECT_ROOT / "configs"
    )
    config_payload = load_stage_config("retrieval", args.config, PROJECT_ROOT / "configs")
    if args.embedding_model:
        config_payload["embedding_model"] = args.embedding_model
    config = RetrievalBuildConfig.model_validate(config_payload)

    parsed_dir = (PROJECT_ROOT / args.parsed_dir).resolve()
    index_dir = (PROJECT_ROOT / args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_retrieval_baseline(
        parsed_dir=parsed_dir,
        index_dir=index_dir,
        config=config,
    )
    manifest_path = index_dir / "index_manifest.json"

    print(f"[Phase3] Loaded retrieval config: {config_path}")
    print(f"[Phase3] Parsed dir: {parsed_dir.as_posix()}")
    print(f"[Phase3] Index dir: {index_dir.as_posix()}")
    print(f"[Phase3] Indexed chunks: {manifest['chunks_indexed']}")
    print(f"[Phase3] Manifest: {manifest_path.as_posix()}")
    print("[Phase3] Build summary:")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
