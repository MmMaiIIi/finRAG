from __future__ import annotations

import json
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from rag_fin.parsing import ParserConfig, parse_pdf_to_artifact, save_parsed_artifact


def _build_sample_pdf(path: Path) -> None:
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "第1页：营收同比增长10%。")

    # Deliberately keep page 2 empty for empty-page handling test.
    _ = doc.new_page()

    page3 = doc.new_page()
    page3.insert_text((72, 72), "第3页：政策支持科技创新。")

    doc.set_metadata({"title": "测试财务政策报告"})
    doc.save(str(path))
    doc.close()


def test_page_numbering_preserved(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _build_sample_pdf(pdf_path)

    artifact = parse_pdf_to_artifact(
        pdf_path,
        ParserConfig(min_text_chars_for_fallback=5),
    )

    assert artifact.page_count == 3
    assert [p.page_num for p in artifact.pages] == [1, 2, 3]


def test_metadata_preserved(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_meta.pdf"
    _build_sample_pdf(pdf_path)

    artifact = parse_pdf_to_artifact(pdf_path, ParserConfig())
    first_page = artifact.pages[0]

    assert artifact.title == "测试财务政策报告"
    assert first_page.doc_id == artifact.doc_id
    assert first_page.metadata["source_path"].endswith("sample_meta.pdf")
    assert first_page.metadata["pdf_metadata"]["title"] == "测试财务政策报告"


def test_empty_page_handling(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_empty.pdf"
    _build_sample_pdf(pdf_path)

    artifact = parse_pdf_to_artifact(pdf_path, ParserConfig())
    empty_page = artifact.pages[1]

    assert empty_page.page_num == 2
    assert empty_page.text == ""
    assert "empty_text_page" in empty_page.parsing_warnings


def test_parsing_artifact_structure(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample_output.pdf"
    _build_sample_pdf(pdf_path)

    artifact = parse_pdf_to_artifact(pdf_path, ParserConfig())
    output_dir = tmp_path / "parsed"
    saved = save_parsed_artifact(
        artifact,
        output_dir=output_dir,
        output_formats=["json", "jsonl", "llamaindex_jsonl"],
    )

    assert set(saved.keys()) == {"json", "jsonl", "llamaindex_jsonl"}

    parsed_json = json.loads(saved["json"].read_text(encoding="utf-8"))
    assert parsed_json["doc_id"] == artifact.doc_id
    assert parsed_json["page_count"] == 3
    assert parsed_json["pages"][0]["page_num"] == 1
    assert "parsing_warnings" in parsed_json["pages"][1]
