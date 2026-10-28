import json

import pymupdf as fitz
import pytest

from akshara.evidence import image_box_to_pdf, read_page
from akshara.meta_client import FakeClient, ResponseError, SPARK, WONDER
from akshara.verify import compare_texts, decide, verify_page


@pytest.mark.parametrize("left,right", [
    ("నమూనా 100 28/10/2026 GO 123/2026", "నమూనా 100 28/10/2026 GO 123/2026"),
    ("రూ. 1,00,000", "రూ. 100000"),
    ("నమూనా\nపత్రం ౧౨", "నమూనా పత్రం 12"),
])
def test_agreement(left, right):
    assert compare_texts(left, right)["agree"]
    assert decide(left, right) == "VERIFIED"


@pytest.mark.parametrize("left,right", [
    ("నమూనా 100", "నమూనా 1000"),
    ("నమూనా 100", "పత్రం 100"),  # Changed name/word cannot pass on numbers.
    ("28/10/2026", "28/11/2026"),
    ("GO 123/2026", "GO 123/2025"),
    ("రూ. -100", "రూ. 100"),
    ("రూ. 100.00", "రూ. 10000"),
    ("₹100", "$100"),
    ("₹100 $200", "$100 ₹200"),
    ("₹100 200", "100 ₹200"),
    ("a 1 b 2", "a 2 b 1"),
    ("", ""),
    ("నమూనా \ufffd", "నమూనా \ufffd"),
])
def test_disagreement(left, right):
    assert not compare_texts(left, right)["agree"]
    assert decide(left, right) == "UNCERTAIN"


def test_second_read_requires_two_independent_reads_to_agree():
    assert decide("amount 100", "amount 1000", "amount 1000") == "CORRECTED"
    assert decide("amount 100", "amount 1000", "amount 100") == "UNCERTAIN"
    assert decide("amount 100", "amount 1000", "amount 2000") == "UNCERTAIN"
    assert decide("amount 100", "", "") == "UNCERTAIN"


def test_independent_reads_never_receive_recovered_text(tmp_path):
    data = dict(spans=[dict(text="image reading 1000", bbox=[10, 10, 500, 100])])
    client = FakeClient([data, data], cache_dir=tmp_path)
    recovered = [dict(page=1, bbox=[1, 1, 100, 30], text="SECRET RECOVERY 100", source="native")]
    with fitz.open() as doc:
        page = doc.new_page()
        spans, evidence = verify_page(client, page, b"png", recovered)
    assert len(client.requests) == 2
    assert [payload["model"] for _, payload in client.requests] == [SPARK, WONDER]
    for _, payload in client.requests:
        assert "SECRET RECOVERY" not in json.dumps(payload)
        assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert evidence["status"] == "UNCERTAIN" and evidence["simulated"]
    assert spans[0]["text"] == "SECRET RECOVERY 100"


def test_page_correction_preserves_original_evidence(tmp_path):
    data = dict(spans=[dict(text="amount 1000", bbox=[10, 10, 500, 100])])
    client = FakeClient([data, data], cache_dir=tmp_path)
    # Exercise real decision behavior without making any API calls.
    client.simulated = False
    with fitz.open() as doc:
        page = doc.new_page()
        recovered = [dict(page=1, bbox=[1, 1, 100, 30], text="amount 100", source="native")]
        spans, evidence = verify_page(client, page, b"png", recovered)
    assert evidence["status"] == "CORRECTED"
    assert evidence["original_spans"][0]["text"] == "amount 100"
    assert spans[0]["text"] == "amount 1000"
    assert spans[0]["source"] == "vision:muse-wonder-1.0"


@pytest.mark.parametrize("box", [[0, 0, 1001, 100], [2, 2, 1, 1], [0, 0, float("nan"), 1], [0, 0, 0, 1], None])
def test_read_rejects_bad_boxes(tmp_path, box):
    client = FakeClient([dict(spans=[dict(text="text", bbox=box)])], cache_dir=tmp_path)
    with fitz.open() as doc:
        page = doc.new_page()
        with pytest.raises(ResponseError):
            read_page(client, page, b"png", model=SPARK, purpose="test")


def test_rotated_image_coordinates_are_pdf_coordinates():
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=100)
        page.set_rotation(90)
        assert image_box_to_pdf(page, [0, 0, 1000, 1000]) == [0, 0, 200, 100]
        assert image_box_to_pdf(page, [0, 0, 500, 500]) == [0, 50, 100, 100]
