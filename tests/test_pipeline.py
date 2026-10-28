import json

import pymupdf as fitz
import pytest

from akshara.__main__ import main
from akshara.meta_client import CacheMiss, FakeClient, MetaClient
from akshara.pipeline import run, segment_page
from akshara.recover import recover_page


@pytest.mark.parametrize("pdf,label,method", [
    ("pdfs2/pstu_music2023.pdf", "unicode_ok", "native"),
    ("pdfs/vsp_1.pdf", "glyph_id", "glyph_id"),
])
def test_real_recovery_pipeline_with_fake_models(tmp_path, survey_root, pdf, label, method):
    client = FakeClient(cache_dir=tmp_path / "cache")
    path = run(survey_root / pdf, client, max_pages=1, output_root=tmp_path / "runs")
    report = json.loads(path.read_text())
    assert report["complete"] and report["simulated"] and report["truncated"]
    assert len(report["pages"]) == 1
    page = report["pages"][0]
    assert page["route"] == label
    assert any(s["source"] == method for s in page["spans"])
    assert any(0xC00 <= ord(c) <= 0xC7F for s in page["spans"] for c in s["text"])
    assert page["verification"]["status"] == "UNCERTAIN"
    assert all(s["verification"] == "UNCERTAIN" for s in page["spans"])
    assert report["client"]["attempted_calls"] == 0
    with fitz.open(survey_root / pdf) as doc:
        bounds = doc[0].rect
        assert all(fitz.Rect(s["bbox"]) in bounds for s in page["spans"])
    assert (path.parent / page["image"]).read_bytes().startswith(b"\x89PNG")
    assert str(survey_root.resolve()) not in path.read_text()
    # Pipeline replay is deterministic and cannot request a network connection.
    replay = FakeClient(cache_dir=tmp_path / "cache", replay=True)
    replay_path = run(survey_root / pdf, replay, max_pages=1, output_root=tmp_path / "replay")
    assert json.loads(replay_path.read_text())["pages"] == report["pages"]
    assert replay.requests == []


def test_structured_records_in_pipeline(tmp_path, survey_root):
    def handler(operation, payload):
        name = payload["response_format"]["json_schema"]["name"]
        if name == "page_read":
            return dict(spans=[])
        evidence = json.loads(payload["messages"][0]["content"][0]["text"].split("EVIDENCE:\n")[1])
        first = evidence[0]
        return dict(records=[dict(kind="office", value=first["text"], cells=[],
                                  evidence=first["text"], span_ids=[0])])
    client = FakeClient(handler=handler, cache_dir=tmp_path / "cache")
    path = run(survey_root / "pdfs2/pstu_music2023.pdf", client, max_pages=1, output_root=tmp_path)
    report = json.loads(path.read_text())
    assert len(report["records"]) == 1
    assert report["records"][0]["verification"] == "UNCERTAIN"
    assert report["records"][0]["bbox"] == report["pages"][0]["spans"][0]["bbox"]


def test_scanned_and_legacy_routes_use_vision(tmp_path, survey_root):
    for pdf, label in [("pdfs2/ap_budget_2025.pdf", "legacy_font"),
                       ("pdfs2/law_telugu_1956.pdf", "scanned")]:
        fake = FakeClient([dict(spans=[dict(text="నమూనా 100", bbox=[10, 20, 900, 50])])],
                          cache_dir=tmp_path / label)
        with fitz.open(survey_root / pdf) as doc:
            spans = recover_page(doc[0], label, b"png", fake)
        assert spans[0]["source"] == "vision:muse-spark-1.3"
        assert spans[0]["text"] == "నమూనా 100"


def test_sam_regions_create_pdf_citations_and_crops(tmp_path):
    def handler(operation, payload):
        concept = payload["input"][0]["content"][0]["text"]
        return dict(boxes=[dict(object_id="0", left=20, top=40, right=100, bottom=80)] if concept == "table" else [], masks=[])
    fake = FakeClient(handler=handler, cache_dir=tmp_path / "cache")
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=300)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        regions = segment_page(fake, page, pix, tmp_path)
    assert len(regions) == 1
    assert regions[0]["bbox"] == [10, 20, 50, 40]
    assert (tmp_path / regions[0]["image"]).is_file()
    assert len(fake.requests) == 4


def test_replay_failure_writes_incomplete_report(tmp_path, survey_root):
    client = MetaClient(tmp_path / "cache", replay=True)
    with pytest.raises(CacheMiss):
        run(survey_root / "pdfs2/pstu_music2023.pdf", client, max_pages=1, output_root=tmp_path)
    report = json.loads((tmp_path / "pstu_music2023/report.json").read_text())
    assert not report["complete"] and report["error"] == "CacheMiss"


def test_cli_triage_and_limits(survey_root, capsys):
    assert main(["triage", str(survey_root / "pdfs2/pstu_music2023.pdf")]) == 0
    assert "document: unicode_ok" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["run", "missing.pdf", "--max-pages", "0", "--fake"])
    assert error.value.code == 2


def test_only_supported_legacy_mappings_are_applied(monkeypatch):
    import akshara.recover as recovery
    def native(font):
        return [dict(page=1, bbox=[0, 0, 10, 10], text="d", font=font,
                     source="native", verification="UNCERTAIN")]
    monkeypatch.setattr(recovery, "native_spans", lambda page: native("Chanakya"))
    mapped = recovery.mapped_legacy(None)
    assert mapped[0]["text"] == "क" and mapped[0]["source"] == "legacy_font"
    monkeypatch.setattr(recovery, "native_spans", lambda page: native("Shree-Tel7-001"))
    assert recovery.mapped_legacy(None) is None
