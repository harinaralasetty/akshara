"""Small per-page port of the M0 survey's font/text rules (no OCR)."""

import csv
import io
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pymupdf as fitz
from fontTools.ttLib import TTFont

LABELS = dict(a="glyph_id", b="legacy_font", c="unicode_ok", d="scanned", e="unclear")
UNICODE = re.compile(
    r"gautami|vani|nirmala|pothana|vemana|mandali|mallanna|ramabhadra|noto.*telugu|"
    r"lohit.*telugu|dhu?rjati|gidugu|gurajada|ntr|peddana|suranna|suravaram|tenali|"
    r"timmana|lakkireddy|sreekrushnadevaraya", re.I,
)
LEGACY = re.compile(
    r"anu(?:script)?|priya(?:nka|anka)|hemalatha|eenadu|vaartha|vaarta|sri[-_ ]?lipi|"
    r"shree|sree|ramana|rachana|pallavi|brahma|kranthi|aaradhana|cvte|telugu[-_ ]?font", re.I,
)


def normalize_font(name):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"^[A-Z]{6}\+", "", name).lower())


def counts(text):
    chars = [c for c in text if not c.isspace()]
    return {
        "chars": len(chars),
        "telugu": sum(0xC00 <= ord(c) <= 0xC7F for c in chars),
        "latin": sum("LATIN" in unicodedata.name(c, "") or 0x21 <= ord(c) <= 0x7E for c in chars),
        "odd": sum(unicodedata.category(c) in ("Co", "Cc", "Cs", "Cn") or c == "\ufffd" for c in chars),
    }


def font_info(doc, entry):
    xref, _, subtype, name, _, encoding, *_ = entry
    info = dict(name=name, subtype=subtype, encoding=encoding, tounicode=False,
                cmap_telugu=0, cmap_low=0, names=[name], cmap_error=None)
    if not xref:
        return info
    info["tounicode"] = doc.xref_get_key(xref, "ToUnicode")[0] != "null"
    data = doc.extract_font(xref)[3]
    if data:
        try:
            with TTFont(io.BytesIO(data), lazy=True) as font:
                tables = font["cmap"].tables if "cmap" in font else []
                codepoints = {cp for t in tables if t.isUnicode() for cp in t.cmap}
                info["cmap_telugu"] = sum(0xC00 <= cp <= 0xC7F for cp in codepoints)
                info["cmap_low"] = sum(0 <= cp <= 255 or 0xF000 <= cp <= 0xF0FF
                                       for t in tables for cp in t.cmap)
                if "name" in font:
                    for n in font["name"].names:
                        if n.nameID in (1, 4, 6):
                            try:
                                info["names"].append(n.toUnicode())
                            except UnicodeError:
                                pass
        except (ValueError, KeyError, AssertionError, OSError) as exc:
            info["cmap_error"] = type(exc).__name__
        except Exception as exc:
            # Invalid embedded SFNTs are unknown, never evidence of a legacy encoding.
            info["cmap_error"] = type(exc).__name__
    return info


def font_label(info, stats):
    n = stats["chars"]
    if n < 20:
        return "unclear"
    te = stats["telugu"] / n
    known_unicode = bool(UNICODE.search(info["name"]))
    legacy_shape = info["subtype"] in ("TrueType", "Type1", "MMType1") or (
        info["subtype"] == "Type0" and info["cmap_telugu"] == 0 and (
            info["cmap_low"] >= 20 or (info["tounicode"] and stats["latin"] / n >= .5)
        )
    )
    if (LEGACY.search(info["name"]) and not known_unicode and legacy_shape
            and te < .1 and (stats["latin"] + stats["odd"]) / n >= .5):
        return "legacy_font"
    if (info["subtype"] == "Type0" and "Identity-" in info["encoding"]
            and not info["tounicode"] and (known_unicode or info["cmap_telugu"] >= 10) and te < .1):
        return "glyph_id"
    if info["tounicode"] and stats["telugu"] >= 20 and te >= .5:
        return "unicode_ok"
    return "unclear"


def choose_label(signals, stats, images, *, document=False):
    if "legacy_font" in signals:
        return "legacy_font"
    if "glyph_id" in signals:
        return "glyph_id"
    if ("unicode_ok" in signals and stats["telugu"] >= (50 if document else 20)
            and stats["telugu"] / max(1, stats["chars"]) >= .01):
        return "unicode_ok"
    if not stats["chars"] and images:
        return "scanned"
    return "unclear"


def triage_document(doc, max_pages=None):
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")
    fonts, totals, pages = {}, defaultdict(list), []
    all_stats = dict(chars=0, telugu=0, latin=0, odd=0)
    for index in range(min(len(doc), max_pages or len(doc))):
        page = doc[index]
        by_name = defaultdict(set)
        for entry in page.get_fonts(full=True):
            xref = entry[0]
            if xref not in fonts:
                fonts[xref] = font_info(doc, entry)
            for name in fonts[xref]["names"]:
                by_name[normalize_font(name)].add(xref)
        page_fonts = defaultdict(list)
        text_dict = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
        for block in text_dict["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    matches = by_name[normalize_font(span["font"])]
                    if len(matches) > 1 and counts(span["text"])["telugu"]:
                        mapped = {x for x in matches if fonts[x]["tounicode"] and fonts[x]["cmap_telugu"] >= 10}
                        matches = mapped or matches
                    for xref in matches:
                        page_fonts[xref].append(span["text"])
                        totals[xref].append(span["text"])
        stats = counts(page.get_text())
        for key in stats:
            all_stats[key] += stats[key]
        images = len(page.get_image_info())  # Includes displayed inline images.
        signals = sorted({font_label(fonts[x], counts(" ".join(t))) for x, t in page_fonts.items()} - {"unclear"})
        pages.append(dict(page=index + 1, label=choose_label(signals, stats, images),
                          signals=signals, **stats, images=images))
    signals = {font_label(fonts[x], counts(" ".join(t))) for x, t in totals.items()} - {"unclear"}
    return dict(label=choose_label(signals, all_stats, sum(p["images"] for p in pages), document=True),
                signals=sorted(signals), pages=pages, **all_stats)


def triage_pdf(path, max_pages=None):
    with fitz.open(path) as doc:
        return triage_document(doc, max_pages)


def check_survey(csv_path, pdf_root):
    categories = {label: dict(correct=0, total=0) for label in LABELS.values()}
    mismatches, errors = [], []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            expected = LABELS[row["classification"]]
            categories[expected]["total"] += 1
            try:
                actual = triage_pdf(Path(pdf_root) / row["file"])["label"]
            except Exception as exc:
                errors.append(dict(file=row["file"], error=type(exc).__name__))
                continue
            if actual == expected:
                categories[expected]["correct"] += 1
            else:
                mismatches.append(dict(file=row["file"], expected=expected, actual=actual))
    for values in categories.values():
        values["agreement"] = values["correct"] / max(1, values["total"])
    correct = sum(c["correct"] for c in categories.values())
    total = sum(c["total"] for c in categories.values())
    return dict(correct=correct, total=total, agreement=correct / max(1, total),
                categories=categories, mismatches=mismatches, errors=errors)
