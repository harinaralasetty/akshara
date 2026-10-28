"""Three bounded live probes; no cache or retries, so each runs exactly once."""

from pathlib import Path

import pymupdf as fitz

from .meta_client import ResponseError, SAM, SPARK


def demo_crop(pdf, page_number=2):
    """Render a small text crop from the public demo, at most 320 x 192 pixels."""
    with fitz.open(pdf) as doc:
        page = doc[page_number - 1]
        blocks = page.get_text("blocks")
        block = next((b for b in blocks if any(0xC00 <= ord(c) <= 0xC7F for c in b[4])), None)
        x, y = (block[0], block[1]) if block else (page.rect.x0, page.rect.y0)
        # Leave one pixel for PyMuPDF's outward rounding of fractional bounds.
        clip = fitz.Rect(x - 4, y - 4, x + 155, y + 91) & page.rect
        return page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False).tobytes("png")


def object_schema(name, kind):
    return dict(type="object", additionalProperties=False, required=[name],
                properties={name: dict(type=kind)})


def smoke(client, pdf=Path("data/demo/ap_2026_I-1.pdf"), *, page=2):
    image = demo_crop(pdf, page)
    probes = [
        ("text", SPARK, lambda: client.chat('Return {"ok": true}.', schema=object_schema("ok", "boolean"),
            schema_name="smoke_text", max_completion_tokens=64), lambda r: r.get("ok") is True),
        ("image", SPARK, lambda: client.chat("Read one visible word. Return JSON with a text string.",
            image=image, schema=object_schema("text", "string"), schema_name="smoke_image",
            max_completion_tokens=64), lambda r: isinstance(r.get("text"), str)),
        ("segment", SAM, lambda: client.segment(image, "text block"),
            lambda r: isinstance(r.get("boxes"), list) and isinstance(r.get("masks"), list)),
    ]
    passed = True
    for name, model, call, parsed in probes:
        start = len(client.usage)
        try:
            result = call()
            ok = isinstance(result, dict) and parsed(result)
        except ResponseError:
            ok = False
        usage = client.usage_summary(start)
        print(f"{name}: model={model} input={usage['input_tokens']} cached={usage['cached_input_tokens']} "
              f"output={usage['output_tokens']} images={usage['images']} "
              f"cost_usd={usage['cost_usd']:.8f} parsing={'ok' if ok else 'FAILED'} "
              f"usage_unreported={usage['unreported_usage_calls']}")
        passed &= ok
    summary = client.usage_summary()
    print(f"Total cost_usd={summary['cost_usd']:.8f}; unresolved_usd={summary['unresolved_usd']:.8f}")
    return passed
