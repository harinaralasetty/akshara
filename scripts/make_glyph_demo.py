#!/usr/bin/env python3
"""Create a SYNTHETIC glyph-ID failure from page 2 of a public Unicode budget."""

import argparse
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akshara.recover import glyph_pages
from akshara.triage import UNICODE, font_info, triage_document, triage_pdf


def make_demo(source="data/demo/ap_2026_I-1.pdf", output_dir="data/demo", *, page=2):
    source = Path(source)
    # These are public unicode_ok documents from the fixed demo allowlist.
    if source.name not in {"ap_2026_I-1.pdf", "ts_legislature.pdf", "ecourts_telugu_2021.pdf"}:
        raise ValueError("Source must be a public Unicode document from the demo allowlist")
    output = Path(output_dir) / f"glyph_demo_SYNTHETIC_{source.stem}.pdf"
    if output.resolve() == source.resolve():
        raise ValueError("Synthetic output must differ from the original")
    with fitz.open(source) as original, fitz.open() as derived:
        if not 1 <= page <= len(original):
            raise ValueError("Source page is out of range")
        derived.insert_pdf(original, from_page=page - 1, to_page=page - 1)
        if triage_document(derived)["label"] != "unicode_ok":
            raise ValueError("Select a page that triages as unicode_ok")
        removed = set()
        for entry in derived[0].get_fonts(full=True):
            info = font_info(derived, entry)
            if (entry[0] and info["tounicode"] and
                    (info["cmap_telugu"] >= 10 or UNICODE.search(info["name"]))):
                derived.xref_set_key(entry[0], "ToUnicode", "null")
                removed.add(entry[0])
        if not removed or triage_document(derived)["label"] != "glyph_id":
            raise ValueError("Removing Telugu ToUnicode maps did not produce glyph_id")
        derived.set_metadata(dict(
            title=f"SYNTHETIC glyph-ID demo: {source.name}, page {page}",
            subject="SYNTHETIC: Telugu ToUnicode maps removed; not an original government PDF",
            creator="Akshara scripts/make_glyph_demo.py", keywords="SYNTHETIC,glyph_id,Telugu"))
        output.parent.mkdir(parents=True, exist_ok=True)
        derived.save(output, garbage=4, deflate=True)
    spans, _ = glyph_pages(output, [1])
    telugu = sum(0xC00 <= ord(c) <= 0xC7F for s in spans.get(1, []) for c in s["text"])
    if triage_pdf(output)["label"] != "glyph_id" or not telugu:
        output.unlink()
        raise ValueError("Synthetic demo failed glyph triage / Telugu recovery validation")
    return output, telugu


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/demo/ap_2026_I-1.pdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--page", type=int, default=2, help="1-based source page (default: 2)")
    args = parser.parse_args(argv)
    try:
        path, telugu = make_demo(args.source, args.output_dir, page=args.page)
        print(f"{path.name}: SYNTHETIC; triage=glyph_id; recovered Telugu characters={telugu}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Glyph demo failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
