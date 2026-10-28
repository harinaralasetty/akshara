#!/usr/bin/env python3
"""Fetch the public demo corpus, verify SHA256, and report actual triage routes."""

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akshara.triage import LABELS, triage_pdf

ALLOWLIST = (
    "pdfs2/ap_2026_I-1.pdf", "pdfs2/ap_2026_v5.pdf", "pdfs2/ap_annex_25-26.pdf",
    "pdfs2/ap_gazette_2014.pdf", "pdfs2/mcr_rti.pdf", "pdfs2/neredcherla_rti.pdf",
    "pdfs2/court_telugu_2024.pdf", "pdfs2/ecourts_telugu_2021.pdf",
    "pdfs2/law_telugu_1956.pdf", "pdfs2/ceots_form6.pdf", "pdfs2/ts_legislature.pdf",
    "pdfs2/archive_dathumala.pdf",
)


def read_manifest(path):
    """Read only public source metadata for the fixed allowlist."""
    selected = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            relative = row["file"]
            if relative not in ALLOWLIST:
                continue
            if relative in selected:
                raise ValueError(f"Duplicate demo document: {relative}")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", row["sha256"]):
                raise ValueError(f"Invalid SHA256: {relative}")
            if not row["url"].startswith("https://"):
                raise ValueError(f"Expected HTTPS source: {relative}")
            selected[relative] = {key: row[key] for key in ("file", "url", "sha256")}
    for relative in ALLOWLIST:
        if relative not in selected:
            raise ValueError(f"Missing demo document: {relative}")
    return [selected[relative] for relative in ALLOWLIST]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_docs(manifest, destination, *, source_root=None):
    """Publish only verified files; an interrupted transfer leaves no partial PDF."""
    rows = read_manifest(manifest)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        target = destination / Path(row["file"]).name
        expected = row["sha256"].lower()
        if not target.is_file() or sha256(target) != expected:
            with tempfile.NamedTemporaryFile(dir=destination, prefix=".fetch-", delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    if source_root is not None:
                        with (Path(source_root) / row["file"]).open("rb") as source:
                            shutil.copyfileobj(source, handle)
                    else:
                        with urlopen(row["url"], timeout=60) as source:
                            shutil.copyfileobj(source, handle)
                    handle.close()
                    if sha256(temporary) != expected:
                        raise ValueError(f"SHA256 mismatch: {row['file']}")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        triage = triage_pdf(target)
        results.append(dict(file=target.name, label=triage["label"], pages=len(triage["pages"])))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    survey_manifest = ROOT / "data/m0b_results.csv"
    default_manifest = survey_manifest if survey_manifest.is_file() else ROOT / "scripts/demo_docs.csv"
    parser.add_argument("--manifest", type=Path, default=default_manifest,
                        help="CSV with file,url,sha256; defaults to local survey CSV or bundled public manifest")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/demo")
    parser.add_argument("--from-local", action="store_true", help="Copy verified sibling corpus files without network")
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "telugu-m0")
    args = parser.parse_args(argv)
    results = fetch_docs(args.manifest, args.output_dir,
                         source_root=args.source_root if args.from_local else None)
    for item in results:
        print(f"{item['file']}: {item['label']} ({item['pages']} pages)")
    coverage = Counter(item["label"] for item in results)
    print(f"Coverage: {len(results)} documents; " + ", ".join(
        f"{label}={coverage[label]}" for label in LABELS.values()))
    missing = [label for label in LABELS.values() if not coverage[label]]
    if missing:
        print("Missing document categories: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"Demo fetch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
