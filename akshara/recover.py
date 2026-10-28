"""Route pages to native extraction, legacylipi, or image reading."""

import pymupdf as fitz

from .evidence import native_spans, page_bounds, read_page
from .meta_client import SPARK


def glyph_pages(pdf, page_numbers):
    """Isolate unsupported glyph content so other pages can still recover."""
    if not page_numbers:
        return {}, []
    from legacylipi.core.glyph_recovery import recover_pdf

    pages, warnings = {}, []
    for number in page_numbers:
        try:
            recovered = recover_pdf(pdf, page_numbers={number}, convert_legacy=False)
        except (ValueError, AssertionError) as exc:
            # Do not include dependency messages that may disclose source paths.
            warnings.append(f"Page {number}: glyph recovery failed ({type(exc).__name__}); using vision")
            continue
        for page in recovered.pages:
            spans = []
            for block in page.text_blocks:
                if block.text.strip() and block.position:
                    box = block.position
                    spans.append(dict(page=page.page_number, bbox=[box.x0, box.y0, box.x1, box.y1],
                                      text=block.text, source="glyph_id" if block.detected_encoding == "glyph-id" else "native",
                                      verification="UNCERTAIN"))
            pages[page.page_number] = spans
        warnings.extend(recovered.warnings)
    return pages, warnings


def mapped_legacy(page):
    """Only font-name matches to packaged mappings; never guess from garbling.

    Current legacylipi has no packaged Telugu legacy mapping. In particular its
    Shree-Dev mapping must not be applied to Shree-Tel fonts.
    """
    from legacylipi.core.encoding_detector import EncodingDetector
    from legacylipi.core.unicode_converter import UnicodeConverter

    detector, converter = EncodingDetector(), UnicodeConverter()
    spans = native_spans(page)
    converted = False
    for span in spans:
        font = span["font"]
        if any(c in font.lower() for c in ("tel", "priya", "pallavi", "anupama")):
            return None
        detection = detector.detect_from_font_name(font)
        if detection and detection.is_legacy:
            result = converter.convert_text(span["text"], detection.detected_encoding)
            if not result.success or result.unmapped_chars:
                return None
            span.update(text=result.converted_text, source="legacy_font")
            converted = True
        elif any(ord(c) > 127 for c in span["text"]):
            return None
    return spans if converted else None


def recover_page(page, label, image, client, *, glyphs=None):
    if label == "unicode_ok":
        return native_spans(page)
    if label == "glyph_id":
        spans = (glyphs or {}).get(page.number + 1)
        if spans:
            for span in spans:
                box = fitz.Rect(span["bbox"]) & page_bounds(page)
                span["bbox"] = list(box)
            return spans
    if label == "legacy_font":
        mapped = mapped_legacy(page)
        if mapped is not None:
            return mapped
    return read_page(client, page, image, model=SPARK, purpose="Recovery read of this page.")
