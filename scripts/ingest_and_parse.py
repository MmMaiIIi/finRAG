from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import fitz

from rag_fin.parsing import (
    ParserConfig,
    list_pdf_files,
    parse_pdf_to_artifact,
    save_parsed_artifact,
    to_llamaindex_node_records,
)
from rag_fin.utils.config import load_stage_config, resolve_config_path
from rag_fin.utils.io import write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 PDF ingestion and parsing."
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
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of PDFs to parse.",
    )
    parser.add_argument(
        "--write-bundle",
        action="store_true",
        help="Write a combined node-ready JSONL bundle for quick inspection.",
    )
    return parser


def _build_parser_config(payload: dict[str, object]) -> ParserConfig:
    return ParserConfig.model_validate(payload)


def _ensure_sample_pdf(input_dir: Path) -> Path:
    """Create a tiny synthetic PDF if the raw directory is empty."""
    input_dir.mkdir(parents=True, exist_ok=True)
    sample_path = input_dir / "phase1_sample.pdf"
    if sample_path.exists():
        return sample_path

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Sample finance report: revenue grew by 10% YoY.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Sample policy note: support technology innovation finance.")
    doc.set_metadata({"title": "Phase1 Synthetic Sample"})
    doc.save(str(sample_path))
    doc.close()
    return sample_path


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_config_path("parser", args.config, PROJECT_ROOT / "configs")
    config_payload = load_stage_config("parser", args.config, PROJECT_ROOT / "configs")
    parser_config = _build_parser_config(config_payload)

    input_dir = (PROJECT_ROOT / args.input_dir).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = list_pdf_files(input_dir)
    if not pdf_paths:
        sample_path = _ensure_sample_pdf(input_dir)
        print(f"[Phase1] No PDFs found. Created sample: {sample_path.as_posix()}")
        pdf_paths = list_pdf_files(input_dir)

    if args.max_files is not None:
        pdf_paths = pdf_paths[: args.max_files]

    all_node_records: list[dict[str, object]] = []
    total_pages = 0
    total_warnings = 0

    print(f"[Phase1] Loaded parser config: {config_path}")
    print(f"[Phase1] Parsing {len(pdf_paths)} PDF(s) from: {input_dir.as_posix()}")
    print(f"[Phase1] Writing artifacts to: {output_dir.as_posix()}")

    for pdf_path in pdf_paths:
        artifact = parse_pdf_to_artifact(pdf_path=pdf_path, config=parser_config)
        saved = save_parsed_artifact(
            artifact=artifact,
            output_dir=output_dir,
            output_formats=parser_config.output_formats,
        )
        node_records = to_llamaindex_node_records(artifact)
        all_node_records.extend(node_records)

        page_warning_count = sum(len(page.parsing_warnings) for page in artifact.pages)
        total_pages += artifact.page_count
        total_warnings += page_warning_count + len(artifact.parsing_warnings)

        saved_labels = ", ".join(f"{k}:{v.name}" for k, v in saved.items())
        print(
            f"[Phase1] Parsed {pdf_path.name} -> pages={artifact.page_count}, "
            f"warnings={page_warning_count}, saved=[{saved_labels}]"
        )

    if args.write_bundle:
        bundle_path = output_dir / "all_pages.llamaindex.jsonl"
        write_jsonl(bundle_path, all_node_records)
        print(f"[Phase1] Wrote bundle: {bundle_path.as_posix()}")

    print(
        f"[Phase1] Done. docs={len(pdf_paths)}, pages={total_pages}, warnings={total_warnings}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
