"""Demo fetching stays within the curated corpus and checks bytes before triage."""

import csv
import hashlib
import importlib.util
import io
from pathlib import Path

import pymupdf as fitz
import pytest


FILES = (
    "pdfs2/ap_2026_I-1.pdf", "pdfs2/ap_2026_v5.pdf", "pdfs2/ap_annex_25-26.pdf",
    "pdfs2/ap_gazette_2014.pdf", "pdfs2/mcr_rti.pdf", "pdfs2/neredcherla_rti.pdf",
    "pdfs2/court_telugu_2024.pdf", "pdfs2/ecourts_telugu_2021.pdf",
    "pdfs2/law_telugu_1956.pdf", "pdfs2/ceots_form6.pdf", "pdfs2/ts_legislature.pdf",
    "pdfs2/archive_dathumala.pdf",
)


@pytest.fixture
def fetcher():
    path = Path(__file__).resolve().parents[1] / "scripts/fetch_demo_docs.py"
    assert path.is_file(), "The demo fetch script must exist"
    spec = importlib.util.spec_from_file_location("fetch_demo_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "source"
    rows = []
    for relative in FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with fitz.open() as doc:
            doc.new_page().insert_text((30, 30), path.stem)
            doc.save(path)
        rows.append(dict(file=relative, url=f"https://example.org/{path.name}",
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(), classification="a"))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return source, manifest


def test_local_fetch_checks_all_hashes_and_triages_actual_bytes(fetcher, corpus, tmp_path, capsys):
    source, manifest = corpus
    destination = tmp_path / "demo"
    assert fetcher.main(["--from-local", "--source-root", str(source),
                         "--manifest", str(manifest), "--output-dir", str(destination)]) == 0
    output = capsys.readouterr().out
    assert output.count("unclear (1 pages)") == 12
    assert "12 documents" in output and "unclear=12" in output and "glyph_id=0" in output
    assert "Missing document categories: glyph_id, legacy_font, unicode_ok, scanned" in output
    assert sorted(p.name for p in destination.iterdir()) == sorted(Path(f).name for f in FILES)
    for relative in FILES:
        assert (destination / Path(relative).name).read_bytes() == (source / relative).read_bytes()


def test_corrupt_source_never_replaces_existing_destination(fetcher, corpus, tmp_path):
    source, manifest = corpus
    destination = tmp_path / "demo"
    destination.mkdir()
    existing = destination / "ap_2026_I-1.pdf"
    existing.write_bytes(b"prior file")
    (source / FILES[0]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetcher.fetch_docs(manifest, destination, source_root=source)
    assert existing.read_bytes() == b"prior file"
    assert list(destination.iterdir()) == [existing]


def test_downloaded_bytes_are_verified_and_cached(fetcher, corpus, tmp_path, monkeypatch):
    source, manifest = corpus
    by_name = {Path(f).name: (source / f).read_bytes() for f in FILES}
    monkeypatch.setattr(fetcher, "urlopen", lambda url, timeout: io.BytesIO(by_name[Path(url).name]))
    destination = tmp_path / "demo"
    fetcher.fetch_docs(manifest, destination)
    def no_download(*args, **kwargs):
        pytest.fail("Verified existing demo files should not be downloaded again")
    monkeypatch.setattr(fetcher, "urlopen", no_download)
    result = fetcher.fetch_docs(manifest, destination)
    assert len(result) == 12
    assert all(item["label"] == "unclear" for item in result)


def test_download_hash_failure_leaves_no_partial_file(fetcher, corpus, tmp_path, monkeypatch):
    _, manifest = corpus
    monkeypatch.setattr(fetcher, "urlopen", lambda url, timeout: io.BytesIO(b"wrong bytes"))
    destination = tmp_path / "demo"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetcher.fetch_docs(manifest, destination)
    assert list(destination.iterdir()) == []


def test_manifest_only_copies_allowlisted_files(fetcher, corpus, tmp_path):
    source, manifest = corpus
    with manifest.open("a") as handle:
        handle.write("../unapproved.pdf,https://example.org/unapproved.pdf," + "0" * 64 + ",a\n")
    result = fetcher.fetch_docs(manifest, tmp_path / "demo", source_root=source)
    assert len(result) == 12


def test_manifest_requires_complete_allowlist(fetcher, corpus, tmp_path):
    source, manifest = corpus
    manifest.write_text(manifest.read_text().replace(FILES[0], "pdfs2/unknown.pdf"))
    with pytest.raises(ValueError, match="Missing demo document"):
        fetcher.fetch_docs(manifest, tmp_path / "demo", source_root=source)
    assert not (tmp_path / "demo").exists()


def test_synthetic_generator_preserves_page_and_labels_web_report(tmp_path, survey_root):
    from fastapi.testclient import TestClient
    from scripts.make_glyph_demo import make_demo
    from akshara.web.app import create_app

    path, telugu = make_demo(survey_root / "ap_2026_I-1.pdf", tmp_path / "data")
    assert "SYNTHETIC" in path.name and telugu > 0
    with fitz.open(path) as derived, fitz.open(survey_root / "ap_2026_I-1.pdf") as source:
        assert all("SYNTHETIC" in derived.metadata[k] for k in ("title", "subject", "keywords"))
        # Preserve drawing commands/fonts; MuPDF rasterization need not be byte-identical
        # after stripping a text map and rebuilding the PDF resource graph.
        assert derived[0].rect == source[1].rect
        assert derived[0].read_contents() == source[1].read_contents()
        assert [derived.extract_font(f[0])[3] for f in derived[0].get_fonts()] == [
            source.extract_font(f[0])[3] for f in source[1].get_fonts()]
    # Renaming cannot hide the metadata disclosure in the web viewer.
    renamed = path.with_name("renamed.pdf")
    renamed.write_bytes(path.read_bytes())
    app = create_app(data_root=renamed.parent, runs_root=tmp_path / "runs", cache_dir=tmp_path / "cache")
    with TestClient(app) as client:
        response = client.post("/api/run", json=dict(path=renamed.name, mode="fake", max_pages=1))
        assert response.status_code == 200, response.text
        report = response.json()["report"]
        assert report["synthetic"] and report["pages"][0]["route"] == "glyph_id"
        assert client.get("/api/documents").json()[0]["synthetic"]
