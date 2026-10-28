"""Independent image reads and deliberately conservative, pure comparison logic."""

import re
import unicodedata

import pymupdf as fitz

from .evidence import read_page
from .meta_client import SPARK, WONDER


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def key_fields(text):
    text = normalize(text)
    # Names cannot be reliably recognized with Telugu regexes. Compare ALL word
    # content as well, so a changed name cannot be excused by matching numbers.
    words = "".join(c if c.isalnum() or unicodedata.category(c).startswith("M")
                    or c in "\u200c\u200d+-−%()" else " " for c in text)
    return dict(
        numbers=sorted(re.findall(r"[+-]?\d+(?:\.\d+)*", text)),
        dates=sorted(re.findall(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b", text)),
        references=sorted(re.findall(r"\b[\w.]*\d+[\w./-]+", text)),
        amounts=sorted(re.sub(r"\s+", "", value) for value in re.findall(
            r"(?:[₹$€£]|usd|inr|rs\.?|రూ\.?)\s*[+-]?\d+(?:\.\d+)*|\d+(?:\.\d+)*\s*[₹$€£]", text)),
        currencies=sorted(c for c in text if unicodedata.category(c) == "Sc"),
        words=words.split(),
    )


def compare_texts(recovered, independent):
    left, right = key_fields(recovered), key_fields(independent)
    matches = {key: left[key] == right[key] for key in left}
    legible = bool(left["words"] and right["words"]) and not any(
        c in recovered + independent for c in ("\ufffd", "\x00"))
    return dict(agree=legible and all(matches.values()), matches=matches,
                recovered=left, independent=right)


def spans_text(spans):
    return "\n".join(span["text"] for span in spans)


def decide(recovered, spark, wonder=None):
    """Two agreeing independent reads can correct recovery; otherwise abstain."""
    first = compare_texts(recovered, spark)
    if first["agree"]:
        return "VERIFIED"
    if wonder is not None and compare_texts(spark, wonder)["agree"]:
        return "CORRECTED"
    return "UNCERTAIN"


def aligned(left, right):
    """Conservatively require one-to-one text and box agreement.

    At least half of each box must overlap. Different span splitting abstains.
    """
    if not left or len(left) != len(right):
        return False
    for a, b in zip(left, right):
        x, y = fitz.Rect(a["bbox"]), fitz.Rect(b["bbox"])
        overlap = (x & y).get_area()
        if (min(x.get_area(), y.get_area()) <= 0
                or overlap < .5 * max(x.get_area(), y.get_area())
                or a["page"] != b["page"]
                or not compare_texts(a["text"], b["text"])["agree"]):
            return False
    return True


def verify_page(client, page, image, recovered):
    spark = read_page(client, page, image, model=SPARK,
                      purpose="First independent verification read. Read only the page image.")
    original_text, spark_text = spans_text(recovered), spans_text(spark)
    comparison = compare_texts(original_text, spark_text)
    wonder = None
    if not comparison["agree"] or not aligned(recovered, spark):
        wonder = read_page(client, page, image, model=WONDER,
                           purpose="Second independent verification read. Read only the page image.")
    status = decide(original_text, spark_text, spans_text(wonder) if wonder is not None else None)
    if ((status == "VERIFIED" and not aligned(recovered, spark))
            or (status == "CORRECTED" and not aligned(spark, wonder))):
        status = "UNCERTAIN"
    selected = wonder if status == "CORRECTED" else recovered
    # A scripted client cannot supply independent verification evidence.
    if client.simulated:
        status, selected = "UNCERTAIN", recovered
    spans = [dict(s, verification=status) for s in selected]
    evidence = dict(status=status, simulated=client.simulated, comparison=comparison,
                    spark_spans=spark, wonder_spans=wonder)
    if status == "CORRECTED":
        evidence["original_spans"] = recovered
    return spans, evidence
