import base64
import io
import json
import subprocess
import sys

import httpx
import pymupdf as fitz
import pytest
from openai import AsyncOpenAI, OpenAI
from PIL import Image

import akshara.meta_client as module
from akshara.__main__ import main


def test_missing_key_exits_one_line_without_pdf_import(tmp_path, monkeypatch):
    monkeypatch.setenv("AKSHARA_ENV_FILE", str(tmp_path / "missing"))
    result = subprocess.run([sys.executable, "-m", "akshara", "smoke"], capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "Smoke skipped: MODEL_API_KEY missing; set it in the environment"]


def test_exactly_three_uncached_tiny_calls_parse_usage_and_sam(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "ap_2026_I-1.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page().insert_text((30, 30), "Public demo text block")
        doc.save(pdf)
    calls = []
    def transport(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["model"] == module.SAM:
            assert payload["input"][0]["content"][0]["text"] == "text block"
            # SAM object records contain both a box and a mask; a bare box is invalid.
            wire = "<0f>0<|box;x1=1;y1=2;x2=2;y2=3;w=10;h=10|><|mask;x=0;y=0;data=2,2,!!!!!&t.SKi|>\n"
            events = [dict(type="response.output_text.delta", item_id="x", output_index=0,
                           content_index=0, delta=wire),
                      dict(type="response.output_text.done", item_id="x", output_index=0,
                           content_index=0, text=wire),
                      dict(type="response.completed", response=dict(id="r", status="completed", output=[],
                           usage=dict(input_tokens=0, output_tokens=0, total_tokens=0)))]
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        assert payload["model"] == module.SPARK and payload["max_completion_tokens"] == 64
        image = len(payload["messages"][0]["content"]) == 2
        content = '{"text":"Public"}' if image else '{"ok":true}'
        return httpx.Response(200, json=dict(id="r", object="chat.completion", created=0, model=module.SPARK,
            choices=[dict(index=0, message=dict(role="assistant", content=content), finish_reason="stop")],
            usage=dict(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                       prompt_tokens_details=dict(cached_tokens=20))))
    monkeypatch.setenv("MODEL_API_KEY", "offline-smoke-key")
    monkeypatch.setattr(module, "OpenAI", lambda **kw: OpenAI(**kw,
        http_client=httpx.Client(transport=httpx.MockTransport(transport))))
    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kw: AsyncOpenAI(**kw,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport))))
    # Run twice: smoke must bypass cache each time.
    for _ in range(2):
        assert main(["smoke", "--pdf", str(pdf)]) == 0
        output = capsys.readouterr().out
        assert output.count("parsing=ok") == 3
        assert "Total cost_usd=0.00279100" in output and "unresolved_usd=0.00000000" in output
    assert len(calls) == 6
    image_url = calls[1]["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url == calls[2]["input"][0]["content"][1]["image_url"]
    with Image.open(io.BytesIO(base64.b64decode(image_url.split(",")[1]))) as image:
        assert image.width <= 320 and image.height <= 192


def test_smoke_parsing_failure_is_reported(tmp_path, capsys):
    from akshara.meta_client import FakeClient
    from akshara.smoke import smoke
    pdf = tmp_path / "demo.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(pdf)
    client = FakeClient([{"ok": False}, {"text": ""}, dict(boxes=[], masks=[])], use_cache=False)
    assert not smoke(client, pdf, page=1)
    assert len(client.requests) == 3
    assert "parsing=FAILED" in capsys.readouterr().out
