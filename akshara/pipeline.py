"""One-document pipeline; all outputs stay in the ignored runs directory."""

import json
from pathlib import Path
from uuid import uuid4

import pymupdf as fitz

from .evidence import image_box_to_pdf, page_bounds
from .extract import extract_records
from .recover import glyph_pages, recover_page
from .triage import triage_document
from .verify import verify_page


def segment_page(client, page, pixmap, output):
    regions = []
    for concept in ("text block", "table", "stamp", "signature"):
        result = client.segment(pixmap.tobytes("png"), concept)
        for box in result["boxes"]:
            pixel_box = [box[k] for k in ("left", "top", "right", "bottom")]
            pdf_box = image_box_to_pdf(page, pixel_box, pixmap.width, pixmap.height)
            ident = f"page-{page.number + 1:03d}-region-{len(regions):03d}"
            # Render a real region crop for every segmentation citation.
            crop = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(pdf_box) * page.rotation_matrix,
                                   alpha=False)
            crop.save(output / f"{ident}.png")
            regions.append(dict(id=ident, concept=concept, bbox=pdf_box, image=f"{ident}.png",
                                source="sam-3.1", object_id=box["object_id"]))
    return regions


def attach_regions(spans, regions):
    for span in spans:
        box = fitz.Rect(span["bbox"])
        span["region_ids"] = [r["id"] for r in regions if (
            (box & fitz.Rect(r["bbox"])).get_area() >= box.get_area() * .5)]


def write_report(output, report, client, usage_start=0):
    report["client"] = client.summary()
    report["usage"] = client.usage_summary(usage_start)
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


def run(pdf, client, *, max_pages=None, output_root="runs"):
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive")
    pdf = Path(pdf)
    output = Path(output_root) / pdf.stem
    output.mkdir(parents=True, exist_ok=True)
    usage_start = len(client.usage)
    report = dict(document=pdf.name, generation=uuid4().hex, complete=False,
                  synthetic="SYNTHETIC" in pdf.name,
                  simulated=client.simulated, pages=[], records=[],
                  coordinates="PDF points, unrotated crop-relative, top-left origin; pages are 1-based",
                  limitations=["Vision boxes are model estimates, not pixel-perfect annotations.",
                               "VERIFIED means conservative reader agreement, not ground truth."])
    write_report(output, report, client, usage_start)
    try:
        with fitz.open(pdf) as doc:
            if doc.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            triage = triage_document(doc, max_pages)
            report["synthetic"] |= "SYNTHETIC" in (doc.metadata.get("subject") or "")
            report.update(page_count=len(doc), selected_pages=len(triage["pages"]), triage=triage,
                          truncated=len(triage["pages"]) < len(doc))
            glyphs, warnings = glyph_pages(pdf, [p["page"] for p in triage["pages"] if p["label"] == "glyph_id"])
            report["recovery_warnings"] = warnings
            for route in triage["pages"]:
                page_usage_start = len(client.usage)
                page = doc[route["page"] - 1]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = pixmap.tobytes("png")
                image_name = f"page-{page.number + 1:03d}.png"
                pixmap.save(output / image_name)
                regions = segment_page(client, page, pixmap, output) if route["label"] == "scanned" else []
                recovered = recover_page(page, route["label"], image, client, glyphs=glyphs)
                spans, verification = verify_page(client, page, image, recovered)
                attach_regions(spans, regions)
                records, rejected = extract_records(client, spans)
                # IDs in records are local to their page's spans array.
                report["pages"].append(dict(page=page.number + 1, route=route["label"], image=image_name,
                                            width=page.rect.width, height=page.rect.height,
                                            bounds=list(page_bounds(page)), rotation=page.rotation,
                                            spans=spans, verification=verification, regions=regions,
                                            usage=client.usage_summary(page_usage_start),
                                            records=records, rejected_records=rejected))
                report["records"].extend(records)
                write_report(output, report, client, usage_start)
        report["complete"] = True
        write_report(output, report, client, usage_start)
    except Exception as exc:
        # Retain completed pages, but never describe an interrupted run as complete.
        report["error"] = type(exc).__name__
        write_report(output, report, client, usage_start)
        raise
    return output / "report.json"
