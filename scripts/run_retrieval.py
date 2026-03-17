from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_fin.indexing import RetrievalBuildConfig, load_index_manifest
from rag_fin.retrieval import run_retrieval, save_retrieval_output
from rag_fin.utils.config import load_stage_config, resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 retrieval baseline.")
    parser.add_argument("query", help="User query text.")
    parser.add_argument("--config", default="default", help="Retrieval config name.")
    parser.add_argument(
        "--index-dir",
        default="data/indexes",
        help="Index directory with dense/BM25 artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "both"],
        default="both",
        help="Retrieval mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to index_dir/retrieval_outputs.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override (must match index build for dense mode).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path(
        "retrieval", args.config, PROJECT_ROOT / "configs"
    )
    config_payload = load_stage_config("retrieval", args.config, PROJECT_ROOT / "configs")
    runtime_cfg = RetrievalBuildConfig.model_validate(config_payload)

    index_dir = (PROJECT_ROOT / args.index_dir).resolve()
    manifest = load_index_manifest(index_dir)
    embedding_model = (
        args.embedding_model
        if args.embedding_model is not None
        else str(manifest.get("embedding_model", runtime_cfg.embedding_model))
    )
    embedding_device = str(
        manifest.get("embedding_device", runtime_cfg.embedding_device)
    )
    mock_embedding_dim = int(
        manifest.get("mock_embedding_dim", runtime_cfg.mock_embedding_dim)
    )
    dense_top_k = int(manifest.get("dense_top_k", runtime_cfg.resolved_dense_top_k))
    bm25_top_k = int(manifest.get("bm25_top_k", runtime_cfg.resolved_bm25_top_k))

    payload = run_retrieval(
        query=args.query,
        index_dir=index_dir,
        mode=args.mode,
        embedding_model_name=embedding_model,
        embedding_device=embedding_device,
        mock_embedding_dim=mock_embedding_dim,
        dense_top_k=dense_top_k,
        bm25_top_k=bm25_top_k,
    )

    output_path = (
        Path(args.output).resolve() if args.output is not None else None
    )
    saved_path = save_retrieval_output(
        payload=payload,
        index_dir=index_dir,
        output_path=output_path,
    )

    print(f"[Phase2] Loaded retrieval config: {config_path}")
    print(f"[Phase2] Index dir: {index_dir.as_posix()}")
    print(f"[Phase2] Mode: {args.mode}")
    print(f"[Phase2] Saved output: {saved_path.as_posix()}")

    for section in ("dense", "bm25"):
        formatted = payload[section]
        print(f"[Phase2] {section} results: {formatted['count']}")
        for item in formatted["results"][:3]:
            print(
                f"  - rank={item['rank']} score={item['score']:.4f} "
                f"citation={item['citation']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
