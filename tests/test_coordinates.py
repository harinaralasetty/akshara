"""Saved page metadata keeps PDF evidence aligned to rotated image pixels."""

import json

import pymupdf as fitz
import pytest

from akshara.meta_client import FakeClient
from akshara.pipeline import run


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_report_preserves_unrotated_bounds_and_rotation(tmp_path, rotation):
    pdf = tmp_path / "rotated.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.set_cropbox(fitz.Rect(10, 20, 190, 280))
        page.set_rotation(rotation)
        doc.save(pdf)
    report_path = run(pdf, FakeClient(cache_dir=tmp_path / "cache"), output_root=tmp_path / "runs")
    page = json.loads(report_path.read_text())["pages"][0]
    assert page["bounds"] == [0, 0, 180, 260]
    assert page["rotation"] == rotation
    assert (page["width"], page["height"]) == ((260, 180) if rotation in (90, 270) else (180, 260))


def test_runs_with_identical_text_metadata_have_distinct_generations(tmp_path):
    from PIL import Image
    import io

    pdf = tmp_path / 'scan.pdf'
    reports, images = [], []
    for index, color in enumerate(('red', 'blue')):
        png = io.BytesIO()
        Image.new('RGB', (20, 30), color).save(png, format='PNG')
        with fitz.open() as doc:
            page = doc.new_page(width=200, height=300)
            page.insert_image(page.rect, stream=png.getvalue())
            doc.save(pdf)
        path = run(pdf, FakeClient(cache_dir=tmp_path / f'cache-{index}'), output_root=tmp_path / f'run-{index}')
        report = json.loads(path.read_text())
        reports.append(report)
        images.append((path.parent / report['pages'][0]['image']).read_bytes())
    assert reports[0]['pages'] == reports[1]['pages']
    assert images[0] != images[1]
    assert reports[0]['generation'] != reports[1]['generation']
