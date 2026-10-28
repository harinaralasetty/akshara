"""Evaluation denominators, usage accounting and cited label scoring stay offline."""

import json

import pymupdf as fitz
import pytest

from akshara.__main__ import main
from akshara.eval import evaluate, metrics, read_labels, score_labels
from akshara.meta_client import FakeClient, MetaClient


def test_metrics_count_pages_statuses_text_and_reported_cost():
    pages = [dict(spans=[dict(text="తెలుగు 100")], verification=dict(status="VERIFIED"),
                  usage=dict(cost_usd=.003, unresolved_usd=.01, unreported_usage_calls=1)),
             dict(spans=[dict(text="   ")], verification=dict(status="UNCERTAIN"), usage={})]
    result = metrics(pages)
    assert result["pages"] == 2 and result["telugu_characters"] == 6
    assert result["pages_with_text"] == 1 and result["text_share"] == .5
    assert result["verification"] == dict(VERIFIED=1, CORRECTED=0, UNCERTAIN=1)
    assert result["cost_per_page_usd"] == .0015
    assert result["unresolved_usd"] == .01 and result["unreported_usage_calls"] == 1
    assert metrics([])["text_share"] == 0


@pytest.fixture
def demo(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    with fitz.open() as pdf:
        pdf.new_page().insert_text((20, 30), "Public budget 100")
        pdf.save(docs / "notice.pdf")
    return docs


def test_fake_cli_writes_report_and_empty_labels_are_not_ground_truth(demo, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(dict(document="notice.pdf", field="amount", question="Amount?", answer="")))
    assert main(["eval", "--mode", "fake", "--docs", str(demo), "--labels", str(labels)]) == 0
    report = tmp_path / "runs/eval/report.md"
    assert report.is_file() and "Not evaluated" in report.read_text()
    result = json.loads((report.parent / "metrics.json").read_text())
    assert len(result["documents"]) == 1 and result["documents"][0]["complete"]
    assert result["routes"]["unclear"]["pages"] == 1
    assert sum(r["pages"] for r in result["routes"].values()) == 1
    assert result["usage"]["cost_usd"] == result["usage"]["paid_calls"] == 0
    assert result["field_accuracy"]["total"] == 0


def test_replay_misses_keep_each_document_and_exit_nonzero(demo, tmp_path, monkeypatch):
    (demo / "bad.pdf").write_bytes(b"not a PDF")
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "--mode", "replay", "--docs", str(demo)]) == 1
    result = json.loads((tmp_path / "runs/eval/metrics.json").read_text())
    assert len(result["documents"]) == 2
    assert all(not d["complete"] for d in result["documents"])
    assert result["documents"][1]["error"] == "CacheMiss"
    assert result["usage"]["paid_calls"] == 0
    assert "Processing errors" in (tmp_path / "runs/eval/report.md").read_text()


def test_live_mode_requires_environment_key_before_writing(demo, tmp_path):
    from akshara.meta_client import MissingKey
    with pytest.raises(MissingKey):
        evaluate(demo, mode="live", runs_root=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_evaluation_assigns_usage_to_document_and_route(demo, tmp_path, monkeypatch):
    # Real MetaClient accounting and cache envelope, with an in-process transport.
    def send(client, payload):
        name = payload["response_format"]["json_schema"]["name"]
        content = dict(spans=[]) if name == "page_read" else dict(records=[])
        raw = dict(choices=[dict(message=dict(content=json.dumps(content)), finish_reason="stop")],
                   usage=dict(prompt_tokens=1000, completion_tokens=100,
                              prompt_tokens_details=dict(cached_tokens=400)))
        reservation = client.guard.reserve(client.estimate(payload))
        client._record(payload, raw, "live", reservation=reservation)
        return raw
    monkeypatch.setattr(MetaClient, "_send_chat", send)
    monkeypatch.setenv("MODEL_API_KEY", "offline-mock-key")
    cache = tmp_path / "cache"
    _, result = evaluate(demo, mode="live", cache_dir=cache, runs_root=tmp_path / "live")
    cost = .001235 * 2 + .004480  # Recovery + Spark verification + Wonder verification.
    assert result["usage"]["cost_usd"] == pytest.approx(cost)
    assert result["documents"][0]["metrics"]["cost_per_page_usd"] == pytest.approx(cost)
    assert result["routes"]["unclear"]["cost_usd"] == pytest.approx(cost)
    monkeypatch.delenv("MODEL_API_KEY")
    def forbidden(*args):
        pytest.fail("Replay attempted to send a request")
    monkeypatch.setattr(MetaClient, "_send_chat", forbidden)
    _, replay = evaluate(demo, mode="replay", cache_dir=cache, runs_root=tmp_path / "replay")
    assert replay["documents"][0]["complete"] and replay["usage"]["cost_usd"] == 0
    assert replay["usage"]["input_tokens"] == 3000


def test_field_accuracy_uses_cited_answers_and_counts_abstention(tmp_path):
    report = tmp_path / "report.json"
    page = dict(page=1, width=200, height=300, spans=[dict(
        text="Budget 100", bbox=[10, 20, 100, 40], verification="UNCERTAIN")])
    report.write_text(json.dumps(dict(document="notice.pdf", records=[], pages=[page])))
    calls = [dict(role="assistant", content=None, tool_calls=[dict(id="read", type="function",
                  function=dict(name="recover_page", arguments='{"page":1}'))]),
             dict(role="assistant", content=json.dumps(dict(abstained=False,
                  citations=[dict(evidence_id="e1", quote="100")])))]
    fake = FakeClient(calls, use_cache=False)
    labels = [dict(document="notice.pdf", field="amount", question="Amount?", answer=" 100\n"),
              dict(document="notice.pdf", field="office", question="Office?", answer="Office"),
              dict(document="missing.pdf", field="date", question="Date?", answer="2026")]
    result = score_labels(labels, {"notice.pdf": report}, fake)
    assert result["correct"] == 1 and result["total"] == 2 and result["accuracy"] == .5
    assert result["by_field"]["amount"] == dict(correct=1, total=1)
    assert result["by_document"]["notice.pdf"] == dict(correct=1, total=2)
    assert result["unscored"] == 1 and result["results"][1]["status"] == "abstained"


@pytest.mark.parametrize("row", [[], dict(document="x", question="?"),
    dict(document="x", question=" ", answer="a"),
    dict(document="x", question="?", answer="a", field=[])])
def test_malformed_labels_fail_clearly(tmp_path, row):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        read_labels(path)


def test_label_template_is_blank_and_covers_public_allowlist():
    from pathlib import Path
    from scripts.fetch_demo_docs import ALLOWLIST
    path = Path(__file__).resolve().parents[1] / "eval/labels.template.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert 15 <= len(rows) <= 20 and all(row["answer"] == "" for row in rows)
    assert {r["document"] for r in rows} == {Path(p).name for p in ALLOWLIST}
