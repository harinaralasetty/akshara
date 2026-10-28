"""Structured extraction whose citations and verification come from evidence."""

import json

import pymupdf as fitz

from .evidence import prompt
from .meta_client import ResponseError

KINDS = ["doc_type", "office", "date", "amount", "reference_number", "table_row"]
RECORD_SCHEMA = dict(type="object", additionalProperties=False, required=["records"], properties=dict(
    records=dict(type="array", items=dict(type="object", additionalProperties=False,
        required=["kind", "value", "cells", "evidence", "span_ids"], properties=dict(
            kind=dict(type="string", enum=KINDS), value=dict(type="string"),
            cells=dict(type="array", items=dict(type="string")), evidence=dict(type="string"),
            span_ids=dict(type="array", items=dict(type="integer"), minItems=1))))))


def whitespace(text):
    return " ".join(text.split())


def ground_records(data, spans):
    """Reject fabricated quotes, values, or span IDs; derive bbox and status locally."""
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ResponseError("Extraction must contain a records array")
    records, rejected = [], []
    for candidate in data["records"]:
        try:
            if not isinstance(candidate, dict) or candidate.get("kind") not in KINDS:
                raise ValueError("invalid record kind")
            ids = candidate["span_ids"]
            if (not isinstance(ids, list) or not ids
                    or any(type(i) is not int or not 0 <= i < len(spans) for i in ids)
                    or len(ids) != len(set(ids)) or ids != sorted(ids)):
                raise ValueError("invalid evidence span IDs")
            support = [spans[i] for i in ids]
            pages = {s["page"] for s in support}
            if len(pages) != 1:
                raise ValueError("citation crosses pages")
            evidence, value, cells = candidate["evidence"], candidate["value"], candidate["cells"]
            if (not isinstance(evidence, str) or not evidence.strip()
                    or not isinstance(value, str) or not value.strip()
                    or not isinstance(cells, list) or any(not isinstance(c, str) or not c.strip() for c in cells)):
                raise ValueError("empty or malformed value/evidence")
            quote = whitespace(evidence)
            source = whitespace(" ".join(s["text"] for s in support))
            if quote not in source:
                raise ValueError("quote absent from cited spans")
            # doc_type is an inferred label; all other values must occur literally.
            if candidate["kind"] != "doc_type" and whitespace(value) not in quote:
                raise ValueError("value absent from quoted evidence")
            if candidate["kind"] == "table_row" and not cells:
                raise ValueError("table row has no cells")
            if any(whitespace(cell) not in quote for cell in cells):
                raise ValueError("table cell absent from quoted evidence")
            box = fitz.Rect(support[0]["bbox"])
            for span in support[1:]:
                box |= fitz.Rect(span["bbox"])
            statuses = {s["verification"] for s in support}
            status = "UNCERTAIN" if "UNCERTAIN" in statuses else (
                "CORRECTED" if "CORRECTED" in statuses else "VERIFIED")
            # An inferred type is not itself independently verified by a literal read.
            if candidate["kind"] == "doc_type" and whitespace(value) not in quote:
                status = "UNCERTAIN"
            records.append(dict(kind=candidate["kind"], value=value, cells=cells, evidence=evidence,
                                span_ids=ids, page=next(iter(pages)), bbox=list(box), verification=status,
                                region_ids=sorted({r for s in support for r in s.get("region_ids", [])})))
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(str(exc))
    return records, rejected


def extract_records(client, spans):
    if not spans:
        return [], []
    evidence = [dict(id=i, text=s["text"], page=s["page"]) for i, s in enumerate(spans)]
    data = client.chat(prompt("extract") + "\nEVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False),
                       schema=RECORD_SCHEMA, schema_name="records")
    return ground_records(data, spans)
