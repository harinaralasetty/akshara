import pymupdf as fitz
import pytest

from akshara.triage import counts, triage_pdf


@pytest.mark.parametrize("relative,label", [
    ("mcr_rti.pdf", "legacy_font"),
    ("ts_legislature.pdf", "unicode_ok"),
    ("law_telugu_1956.pdf", "scanned"),
    ("ceots_form6.pdf", "unclear"),
])
def test_survey_categories(survey_root, relative, label):
    result = triage_pdf(survey_root / relative)
    assert result["label"] == label
    assert any(p["label"] == label for p in result["pages"])
    assert [p["page"] for p in result["pages"]] == list(range(1, len(result["pages"]) + 1))


def test_blank_outlines_and_english_are_not_scans(tmp_path):
    pdf = tmp_path / "vector.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.draw_rect(fitz.Rect(20, 20, 100, 100))
        page = doc.new_page()
        page.insert_text((20, 20), "English text is not evidence of Telugu legacy encoding.")
        doc.save(pdf)
    result = triage_pdf(pdf)
    assert result["label"] == "unclear"
    assert all(p["label"] == "unclear" for p in result["pages"])


def test_tounicode_alone_does_not_mean_unicode(survey_root):
    result = triage_pdf(survey_root / "mcr_rti.pdf", max_pages=1)
    assert result["pages"][0]["label"] == "legacy_font"
    assert result["telugu"] == 0


def test_counts_exclude_whitespace():
    assert counts(" అ 1\n\t\ufffd") == dict(chars=3, telugu=1, latin=1, odd=1)
