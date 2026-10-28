"""Offline/replay/live evaluation of selected demo pages and optional labelled QA."""

import json
import unicodedata
from collections import Counter
from pathlib import Path

from .agent import ask
from .meta_client import FakeClient, MetaClient
from .pipeline import run
from .triage import LABELS, counts


def metrics(pages):
    statuses = Counter(p["verification"]["status"] for p in pages)
    with_text = sum(any(s["text"].strip() for s in p["spans"]) for p in pages)
    cost = sum(p.get("usage", {}).get("cost_usd", 0) for p in pages)
    return dict(pages=len(pages), telugu_characters=sum(
        counts(s["text"])["telugu"] for p in pages for s in p["spans"]),
        pages_with_text=with_text, text_share=with_text / len(pages) if pages else 0,
        verification={s: statuses[s] for s in ("VERIFIED", "CORRECTED", "UNCERTAIN")},
        cost_usd=round(cost, 12), cost_per_page_usd=cost / len(pages) if pages else 0,
        unresolved_usd=sum(p.get("usage", {}).get("unresolved_usd", 0) for p in pages),
        unreported_usage_calls=sum(p.get("usage", {}).get("unreported_usage_calls", 0) for p in pages))


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_labels(path):
    path = Path(path)
    if not path.is_file():
        return []
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
                not isinstance(row.get(k), str) for k in ("document", "question", "answer")):
            raise ValueError("Labels need document, question and answer strings")
        if not row["document"].strip() or not row["question"].strip():
            raise ValueError("Label document and question must not be blank")
        if "field" in row and (not isinstance(row["field"], str) or not row["field"].strip()):
            raise ValueError("Label field must be a nonempty string")
        if row["answer"].strip():
            labels.append(row)
    return labels


def score_labels(labels, reports, client):
    """Exact match of cited extractive QA; empty answer slots are never ground truth."""
    rows = []
    for label in labels:
        path = reports.get(label["document"])
        row = dict(document=label["document"], field=label.get("field", "unspecified"),
                   question=label["question"], correct=False, status="document_not_evaluated")
        if path is not None:
            try:
                answer = ask(path, label["question"], client)
                row.update(status="abstained" if answer["abstained"] else "answered",
                           correct=not answer["abstained"] and
                           normalized(answer["answer"]) == normalized(label["answer"]))
            except Exception as exc:
                row["status"] = type(exc).__name__
        rows.append(row)
    scored = [r for r in rows if r["status"] != "document_not_evaluated"]
    correct = sum(r["correct"] for r in scored)
    fields, documents = {}, {}
    for row in scored:
        for group, key in ((fields, row["field"]), (documents, row["document"])):
            item = group.setdefault(key, dict(correct=0, total=0))
            item["total"] += 1
            item["correct"] += row["correct"]
    return dict(correct=correct, total=len(scored), accuracy=correct / len(scored) if scored else None,
                by_field=fields, by_document=documents, results=rows, unscored=len(rows) - len(scored))


def markdown(result):
    lines = ["# Akshara evaluation", "", f"Mode: **{result['mode']}**. "
             f"Page limit per document: **{result['max_pages']}** (from page 1).", "",
             "Counts cover completed selected pages, not the entire documents. "
             "Telugu characters are Unicode code points, not a reading-accuracy score. "
             "Verification counts are page counts; agreement is not ground truth.", "",
             "Fake model reads abstain; local native/glyph recovery is real. "
             "Fake and replay incur no API charge. Costs use response usage and the configured tariff; "
             "unreported usage and failed attempts retain separate unresolved reservations. "
             "USD/page covers completed pipeline pages; usage totals also include failed page attempts "
             "and labelled question answering.", "",
             "## Per document", "",
             "| Document | Complete | Completed / total | Pages per route | Telugu chars | Pages with text | VERIFIED / CORRECTED / UNCERTAIN | USD / page |",
             "| --- | --- | --- | --- | ---: | ---: | --- | ---: |"]
    for doc in result["documents"]:
        m = doc["metrics"]
        routes = ", ".join(f"{k}: {v}" for k, v in doc["pages_per_route"].items()) or "none"
        status = " / ".join(str(v) for v in m["verification"].values())
        name = doc["document"] + (" (SYNTHETIC)" if doc["synthetic"] else "")
        lines.append(f"| {name} | {doc['complete']} | {m['pages']} / {doc['page_count']} | {routes} | "
                     f"{m['telugu_characters']} | {m['pages_with_text']}/{m['pages']} ({m['text_share']:.1%}) | "
                     f"{status} | {m['cost_per_page_usd']:.8f} |")
    errors = [f"- {d['document']}: {d['error']}" for d in result["documents"] if d.get("error")]
    if errors:
        lines += ["", "Processing errors (completed pages are retained):", "", *errors]
    lines += ["", "## Per route", "",
              "| Route | Pages | Telugu chars | Pages with text | VERIFIED / CORRECTED / UNCERTAIN | USD / page |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
    for route, m in result["routes"].items():
        status = " / ".join(str(v) for v in m["verification"].values())
        lines.append(f"| {route} | {m['pages']} | {m['telugu_characters']} | "
                     f"{m['pages_with_text']}/{m['pages']} ({m['text_share']:.1%}) | "
                     f"{status} | {m['cost_per_page_usd']:.8f} |")
    usage = result["usage"]
    lines += ["", "## Usage", "",
              "| Model | Calls | Paid attempts | Input tokens | Cached input | Output tokens | Images | USD |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, u in usage["by_model"].items():
        lines.append(f"| {model} | {u['calls']} | {u['paid_calls']} | {u['input_tokens']} | "
                     f"{u['cached_input_tokens']} | {u['output_tokens']} | {u['images']} | {u['cost_usd']:.8f} |")
    lines += ["", f"Total measured USD: {usage['cost_usd']:.8f}; "
              f"unresolved reservation USD: {usage['unresolved_usd']:.8f}; "
              f"calls without token usage: {usage['unreported_usage_calls']}.", "", "## Field accuracy", "",
              "Exact match of cited question answers, grouped by labelled field. "
              "NFC and whitespace are normalized; no partial-credit or semantic matching. "
              "Questions should target the selected pages; unprocessed-document labels are unscored. "
              "Empty answer slots are ignored.", ""]
    accuracy = result["field_accuracy"]
    if accuracy["accuracy"] is None:
        lines.append("Not evaluated: no filled labels for evaluated documents. "
                     "Copy eval/labels.template.jsonl to eval/labels.jsonl for a Telugu reader to annotate.")
    else:
        lines.append(f"Correct: {accuracy['correct']}/{accuracy['total']} ({accuracy['accuracy']:.1%}); "
                     f"unscored labels: {accuracy['unscored']}.")
        for field, row in accuracy["by_field"].items():
            lines.append(f"- {field}: {row['correct']}/{row['total']}")
        lines += ["", "| Document | Correct / labelled |", "| --- | ---: |"]
        for document, row in accuracy["by_document"].items():
            lines.append(f"| {document} | {row['correct']}/{row['total']} |")
    if accuracy["results"]:
        lines += ["", "| Document | Field | Result |", "| --- | --- | --- |"]
        for row in accuracy["results"]:
            status = "correct" if row["correct"] else row["status"]
            if status == "answered":
                status = "incorrect"
            lines.append(f"| {row['document']} | {row['field']} | {status} |")
    return "\n".join(lines) + "\n"


def evaluate(docs="data/demo", *, mode="fake", max_pages=2, cap=3, cache_dir="cache",
             labels="eval/labels.jsonl", runs_root="runs", client=None):
    if mode not in {"fake", "replay", "live"}:
        raise ValueError("Mode must be fake, replay or live")
    if type(max_pages) is not int or max_pages <= 0:
        raise ValueError("max_pages must be a positive integer")
    pdfs = sorted(Path(docs).glob("*.pdf"))
    if not pdfs:
        raise ValueError("No demo PDFs found; prepare data/demo first")
    annotations = read_labels(labels)
    if client is None:
        client = (FakeClient if mode == "fake" else MetaClient)(
            cache_dir=cache_dir, replay=mode == "replay", cap=cap)
    if mode == "live":
        client._key()
    start = len(client.usage)
    documents, all_pages, reports = [], [], {}
    for pdf in pdfs:
        path = Path(runs_root) / pdf.stem / "report.json"
        try:
            run(pdf, client, max_pages=max_pages, output_root=runs_root)
        except Exception as exc:
            # The pipeline retains a partial report; a bad PDF must not hide peers.
            if not path.is_file():
                documents.append(dict(document=pdf.name, complete=False, synthetic="SYNTHETIC" in pdf.name,
                    page_count=0, pages_per_route={}, metrics=metrics([]), error=type(exc).__name__))
                continue
        report = json.loads(path.read_text(encoding="utf-8"))
        pages = report["pages"]
        all_pages.extend(pages)
        documents.append(dict(document=pdf.name, complete=report["complete"], synthetic=report["synthetic"],
            page_count=report.get("page_count", 0), pages_per_route=dict(Counter(p["route"] for p in pages)),
            metrics=metrics(pages), usage=report["usage"], error=report.get("error")))
        reports[pdf.name] = path
    accuracy = score_labels(annotations, reports, client)
    result = dict(mode=mode, max_pages=max_pages, documents=documents,
        routes={route: metrics([p for p in all_pages if p["route"] == route]) for route in LABELS.values()},
        field_accuracy=accuracy, usage=client.usage_summary(start))
    output = Path(runs_root) / "eval"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(markdown(result), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output / "report.md", result
