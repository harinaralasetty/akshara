"""Shared evidence shapes and image-to-PDF coordinates (1-based pages)."""

import math
from pathlib import Path

import pymupdf as fitz

from .meta_client import ResponseError

BBOX_SCHEMA = dict(type="array", items=dict(type="number"), minItems=4, maxItems=4)
READ_SCHEMA = dict(
    type="object", additionalProperties=False, required=["spans"], properties=dict(spans=dict(
        type="array", items=dict(type="object", additionalProperties=False,
            required=["text", "bbox"], properties=dict(text=dict(type="string"), bbox=BBOX_SCHEMA)))))


def prompt(name):
    return (Path(__file__).with_name("prompts") / f"{name}.txt").read_text(encoding="utf-8")


def page_bounds(page):
    return page.rect * page.derotation_matrix


def valid_bbox(box, bounds):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
        raise ResponseError("Invalid bounding box")
    x0, y0, x1, y1 = box
    if not (bounds[0] <= x0 < x1 <= bounds[2] and bounds[1] <= y0 < y1 <= bounds[3]):
        raise ResponseError("Bounding box outside page or empty")
    return list(box)


def image_box_to_pdf(page, box, width=1000, height=1000):
    valid_bbox(box, [0, 0, width, height])
    rect = fitz.Rect(box)
    rect = fitz.Rect(rect.x0 / width * page.rect.width, rect.y0 / height * page.rect.height,
                     rect.x1 / width * page.rect.width, rect.y1 / height * page.rect.height)
    return list(rect * page.derotation_matrix)


def native_spans(page):
    result = []
    for block in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    box = fitz.Rect(span["bbox"]) & page_bounds(page)
                    if not box.is_empty:
                        result.append(dict(page=page.number + 1, bbox=list(box), text=span["text"],
                                           source="native", font=span["font"], verification="UNCERTAIN"))
    return result


def read_page(client, page, image, *, model, purpose):
    data = client.chat(prompt("read") + "\n" + purpose, image=image, schema=READ_SCHEMA,
                       schema_name="page_read", model=model)
    if not isinstance(data, dict) or not isinstance(data.get("spans"), list):
        raise ResponseError("Page read must contain a spans array")
    result = []
    for span in data["spans"]:
        if not isinstance(span, dict) or not isinstance(span.get("text"), str):
            raise ResponseError("Invalid page-read span")
        box = image_box_to_pdf(page, span.get("bbox"))
        if span["text"].strip():
            result.append(dict(page=page.number + 1, bbox=box, text=span["text"],
                               source=f"vision:{model}", verification="UNCERTAIN"))
    return result
