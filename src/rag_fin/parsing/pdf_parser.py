from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
from pydantic import BaseModel, ConfigDict, Field

from rag_fin.schemas import ParsedDocumentArtifact, ParsedPageRecord
from rag_fin.utils.io import write_jsonl


class ParserConfig(BaseModel):
    """Config for PDF parsing behavior."""

    model_config = ConfigDict(extra="forbid")

    pdf_backend_primary: str = "pymupdf"
    pdf_backend_table: str = "pdfplumber"
    enable_pdfplumber_fallback: bool = True
    prefer_pdfplumber_for_table_like_pages: bool = True
    min_text_chars_for_fallback: int = Field(default=20, ge=0)
    preserve_page_level_provenance: bool = True
    output_formats: list[str] = Field(default_factory=lambda: ["json", "jsonl"])


def list_pdf_files(input_dir: str | Path) -> list[Path]:
    """List PDF files under an input directory."""
    directory = Path(input_dir)
    if not directory.exists():
        return []
    return sorted([path for path in directory.glob("*.pdf") if path.is_file()])


def clean_text(text: str) -> str:
    """Normalize extracted text for inspectable page-level artifacts."""
    normalized = text.replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _derive_doc_id(pdf_path: Path) -> str:
    doc_id = re.sub(r"[^0-9A-Za-z_\-]+", "_", pdf_path.stem)
    return doc_id.strip("_") or "document"


def _page_title(pdf_title: str | None, pdf_path: Path) -> str:
    if pdf_title and pdf_title.strip():
        return pdf_title.strip()
    return pdf_path.stem


def _build_page_metadata(
    *,
    doc_id: str,
    title: str,
    source_path: Path,
    page_num: int,
    parser_used: str,
    pdf_metadata: dict[str, str],
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "title": title,
        "source_path": str(source_path.as_posix()),
        "page_num": page_num,
        "parser_used": parser_used,
        "pdf_metadata": pdf_metadata,
    }


def _extract_with_pdfplumber(
    plumber_pdf: pdfplumber.pdf.PDF | None,
    page_index: int,
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if plumber_pdf is None:
        warnings.append("pdfplumber_not_available_for_page")
        return "", warnings

    try:
        text = plumber_pdf.pages[page_index].extract_text() or ""
        return clean_text(text), warnings
    except Exception as exc:  # pragma: no cover - defensive branch
        warnings.append(f"pdfplumber_extract_failed:{type(exc).__name__}")
        return "", warnings


def parse_pdf_to_artifact(
    pdf_path: str | Path,
    config: ParserConfig,
) -> ParsedDocumentArtifact:
    """Parse a single PDF into page-level inspectable artifact records."""
    source_path = Path(pdf_path).resolve()
    doc_warnings: list[str] = []

    if config.pdf_backend_primary.lower() != "pymupdf":
        raise ValueError("Phase 1 parser expects pdf_backend_primary='pymupdf'")

    with fitz.open(source_path) as pdf:
        raw_meta = pdf.metadata or {}
        safe_meta = {k: v for k, v in raw_meta.items() if isinstance(v, str) and v}
        title = _page_title(safe_meta.get("title"), source_path)
        doc_id = _derive_doc_id(source_path)

        plumber_pdf: pdfplumber.pdf.PDF | None = None
        if config.enable_pdfplumber_fallback:
            try:
                plumber_pdf = pdfplumber.open(str(source_path))
            except Exception as exc:  # pragma: no cover - defensive branch
                doc_warnings.append(f"pdfplumber_open_failed:{type(exc).__name__}")

        pages: list[ParsedPageRecord] = []
        for page_index, page in enumerate(pdf, start=0):
            warnings: list[str] = []
            page_num = page_index + 1

            text_pymupdf = clean_text(page.get_text("text"))
            text = text_pymupdf
            parser_used = "pymupdf"

            should_try_plumber = (
                config.enable_pdfplumber_fallback
                and len(text_pymupdf) < config.min_text_chars_for_fallback
            )
            if should_try_plumber:
                text_plumber, plumber_warnings = _extract_with_pdfplumber(
                    plumber_pdf, page_index
                )
                warnings.extend(plumber_warnings)

                if text_plumber and (
                    not text_pymupdf
                    or (
                        config.prefer_pdfplumber_for_table_like_pages
                        and len(text_plumber) > len(text_pymupdf)
                    )
                ):
                    text = text_plumber
                    parser_used = "pdfplumber"
                    warnings.append("pdfplumber_text_selected")

            if not text:
                warnings.append("empty_text_page")

            page_meta = _build_page_metadata(
                doc_id=doc_id,
                title=title,
                source_path=source_path,
                page_num=page_num,
                parser_used=parser_used,
                pdf_metadata=safe_meta,
            )
            pages.append(
                ParsedPageRecord(
                    doc_id=doc_id,
                    title=title,
                    source_path=str(source_path.as_posix()),
                    page_num=page_num,
                    text=text,
                    metadata=page_meta,
                    parsing_warnings=warnings,
                )
            )

        if plumber_pdf is not None:
            plumber_pdf.close()

    return ParsedDocumentArtifact(
        doc_id=doc_id,
        title=title,
        source_path=str(source_path.as_posix()),
        page_count=len(pages),
        metadata={"pdf_metadata": safe_meta},
        parsing_warnings=doc_warnings,
        pages=pages,
    )


def to_llamaindex_node_records(
    artifact: ParsedDocumentArtifact,
) -> list[dict[str, Any]]:
    """Return node-ready records that map directly into LlamaIndex Document objects."""
    records: list[dict[str, Any]] = []
    for page in artifact.pages:
        records.append(
            {
                "id": f"{artifact.doc_id}:p{page.page_num}",
                "text": page.text,
                "metadata": {
                    "doc_id": page.doc_id,
                    "title": page.title,
                    "source_path": page.source_path,
                    "page_num": page.page_num,
                    "parsing_warnings": page.parsing_warnings,
                    **page.metadata,
                },
            }
        )
    return records


def save_parsed_artifact(
    artifact: ParsedDocumentArtifact,
    output_dir: str | Path,
    output_formats: list[str] | None = None,
) -> dict[str, Path]:
    """Save human-readable parsed outputs for inspection and downstream stages."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = output_formats or ["json", "jsonl"]
    saved: dict[str, Path] = {}

    if "json" in formats:
        json_path = out_dir / f"{artifact.doc_id}.parsed.json"
        json_path.write_text(
            artifact.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        saved["json"] = json_path

    if "jsonl" in formats:
        pages_path = out_dir / f"{artifact.doc_id}.pages.jsonl"
        page_records = [page.model_dump(mode="json") for page in artifact.pages]
        write_jsonl(pages_path, page_records)
        saved["jsonl"] = pages_path

    if "llamaindex_jsonl" in formats:
        llamaindex_path = out_dir / f"{artifact.doc_id}.llamaindex.jsonl"
        write_jsonl(llamaindex_path, to_llamaindex_node_records(artifact))
        saved["llamaindex_jsonl"] = llamaindex_path

    return saved
