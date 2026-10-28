"""Offline HTTP contract and contained filesystem access for the local demo."""

import json
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from akshara.meta_client import FakeClient


@pytest.fixture
def web(tmp_path):
    from akshara.web.app import create_app

    data = tmp_path / "data"
    (data / "demo").mkdir(parents=True)
    pdf = data / "demo" / "notice.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 30), "Public notice 2026")
        doc.save(pdf)
    app = create_app(data_root=data, runs_root=tmp_path / "runs", cache_dir=tmp_path / "cache")
    with TestClient(app) as client:
        yield client, data, tmp_path / "runs"


def test_fake_run_report_image_and_ask_flow(web):
    client, _, _ = web
    assert client.get("/api/documents").json() == []
    result = client.post("/api/run", json={"path": "demo/notice.pdf", "mode": "fake", "max_pages": 1})
    assert result.status_code == 200, result.text
    assert result.json()["document"] == "notice"
    documents = client.get("/api/documents").json()
    assert documents[0]["id"] == "notice"
    report = client.get("/api/documents/notice/report").json()
    assert report["complete"] and report["simulated"]
    assert report["pages"][0]["verification"]["status"] == "UNCERTAIN"
    assert report["pages"][0]["bounds"] == [0, 0, 200, 300]
    image = client.get("/api/documents/notice/pages/1/image")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG")
    answer = client.post("/api/ask", json={"document": "notice", "question": "What is the notice?", "mode": "fake"})
    assert answer.status_code == 200, answer.text
    assert answer.json()["abstained"] and not answer.json()["citations"]
    assert answer.json()["simulated"]
    assert "cannot verify" in answer.json()["answer"].lower()


def test_single_page_contains_required_ui_and_author(web):
    client, _, _ = web
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    for value in ('name="author"', 'By Hari Naralasetty', 'https://github.com/harinaralasetty',
                  'https://www.linkedin.com/in/harinaralasetty', 'VERIFIED', 'CORRECTED', 'UNCERTAIN'):
        assert value in html
    assert '<script src=' not in html and '<link rel="stylesheet"' not in html


@pytest.mark.parametrize("path", ["../outside.pdf", "demo/notice.txt", "missing.pdf"])
def test_run_rejects_invalid_or_external_pdf(web, path):
    client, _, runs = web
    response = client.post("/api/run", json={"path": path, "mode": "fake"})
    assert response.status_code in (400, 404)
    assert not runs.exists()


def test_symlinks_cannot_escape_data_or_runs(web, tmp_path):
    client, data, runs = web
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    assert client.post('/api/run', json={'path': str(outside), 'mode': 'fake'}).status_code == 400
    (data / "linked.pdf").symlink_to(outside)
    assert client.post("/api/run", json={"path": "linked.pdf", "mode": "fake"}).status_code == 400
    runs.mkdir()
    (runs / "linked").symlink_to(tmp_path, target_is_directory=True)
    assert client.get("/api/documents/linked/report").status_code == 400
    assert client.get("/api/documents").json() == []


def test_missing_report_and_page_return_404(web):
    client, _, _ = web
    assert client.get("/api/documents/missing/report").status_code == 404
    assert client.post("/api/ask", json={"document": "missing", "question": "What?"}).status_code == 404
    client.post("/api/run", json={"path": "demo/notice.pdf", "mode": "fake"})
    assert client.get("/api/documents/notice/pages/2/image").status_code == 404


def test_replay_cache_miss_is_reported_without_network(web):
    client, _, _ = web
    response = client.post("/api/run", json={"path": "demo/notice.pdf", "mode": "replay"})
    assert response.status_code == 409
    report = client.get("/api/documents/notice/report").json()
    assert not report["complete"] and report["error"] == "CacheMiss"


@pytest.mark.parametrize("body", [
    {"path": "demo/notice.pdf", "mode": "wrong"},
    {"path": "demo/notice.pdf", "max_pages": 0},
])
def test_run_validates_request(web, body):
    assert web[0].post("/api/run", json=body).status_code == 422


def test_ask_validates_question_and_step_limit(web):
    client, _, _ = web
    for extra in ({"question": " "}, {"question": "What?", "max_steps": 0}):
        assert client.post("/api/ask", json={"document": "notice", **extra}).status_code == 422


def test_report_image_must_be_inside_document_directory(web, tmp_path):
    client, _, runs = web
    directory = runs / "bad"
    directory.mkdir(parents=True)
    (tmp_path / "private.png").write_bytes(b"private")
    (directory / "report.json").write_text(json.dumps({"pages": [{"page": 1, "image": "../../private.png"}]}))
    assert client.get("/api/documents/bad/pages/1/image").status_code == 400


def test_scripted_cited_answer_through_http(tmp_path):
    from akshara.web.app import create_app

    runs = tmp_path / "runs"
    directory = runs / "budget"
    directory.mkdir(parents=True)
    evidence = dict(page=1, bbox=[10, 20, 100, 40], text="Budget Rs. 100", verification="UNCERTAIN")
    record = dict(evidence, kind="amount", value="100", evidence="Budget Rs. 100", cells=[])
    (directory / "report.json").write_text(json.dumps(dict(simulated=True, records=[record], pages=[
        dict(page=1, bounds=[0, 0, 200, 300], width=200, height=300, spans=[evidence], regions=[])])))
    messages = [
        dict(role="assistant", content=None, tool_calls=[dict(id="search", type="function", function=dict(
            name="search_records", arguments=json.dumps({"query": "Budget"})))]),
        dict(role="assistant", content=json.dumps({"answer": "Budget Rs. 100", "citations": [
            {"page": 1, "bbox": [10, 20, 100, 40], "quote": "Budget Rs. 100"}]})),
    ]
    fake = FakeClient(messages, cache_dir=tmp_path / "cache")
    app = create_app(data_root=tmp_path / "data", runs_root=runs, client_factory=lambda mode: fake)
    with TestClient(app) as client:
        response = client.post("/api/ask", json={"document": "budget", "question": "Budget?", "mode": "fake"})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert not answer["abstained"]
    assert "100" in answer["answer"]
    assert answer["citations"][0]["bbox"] == [10, 20, 100, 40]


def test_serve_cli_dispatches_to_local_uvicorn(monkeypatch):
    import uvicorn
    from akshara.__main__ import main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    assert main(["serve", "--port", "8123"]) == 0
    assert calls == [("akshara.web.app:app", {"host": "127.0.0.1", "port": 8123})]


def test_ask_replay_miss_is_an_operational_error(web):
    client, _, _ = web
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    response = client.post('/api/ask', json={'document': 'notice', 'question': 'What?', 'mode': 'replay'})
    assert response.status_code == 409
    assert 'Replay cache miss' in response.json()['detail']


def test_rerun_rejects_stale_report_image_revision(web):
    client, data, _ = web
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    first = client.get('/api/documents/notice/report').json()
    revision = first['revision']
    assert client.get('/api/documents/notice/pages/1/image', params={'revision': revision}).status_code == 200
    with fitz.open() as doc:
        doc.new_page(width=250, height=200).insert_text((20, 30), 'Replaced public notice')
        doc.save(data / 'demo/notice.pdf')
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    assert client.get('/api/documents/notice/report').json()['revision'] != revision
    assert client.get('/api/documents/notice/pages/1/image', params={'revision': revision}).status_code == 409


def test_run_rejects_output_image_symlink_escape(web, tmp_path):
    client, _, runs = web
    directory = runs / 'notice'
    directory.mkdir(parents=True)
    protected = tmp_path / 'outside.png'
    protected.write_bytes(b'preserve')
    (directory / 'page-001.png').symlink_to(protected)
    assert client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'}).status_code == 400
    assert protected.read_bytes() == b'preserve'


def test_ask_rejects_an_outdated_report_revision(web):
    client, _, _ = web
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    response = client.post('/api/ask', json={
        'document': 'notice', 'question': 'What?', 'revision': 'outdated', 'mode': 'fake'})
    assert response.status_code == 409
    assert 'changed' in response.json()['detail']


def test_processing_blocks_competing_writes_and_image_reads(web, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from importlib import import_module
    from threading import Event

    client, _, _ = web
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    module = import_module('akshara.web.app')
    original = module.run
    started, release = Event(), Event()
    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5), 'Test must release the processing call'
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'run', blocked)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, '/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
        try:
            assert started.wait(5)
            assert client.get('/api/documents/notice/report').status_code == 409
            assert client.get('/api/documents/notice/pages/1/image').status_code == 409
            assert client.post('/api/ask', json={'document': 'notice', 'question': 'What?'}).status_code == 409
            assert client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'}).status_code == 409
        finally:
            release.set()
        assert pending.result().status_code == 200
    assert client.get('/api/documents/notice/pages/1/image').status_code == 200


def test_demo_document_fake_run_report_and_ask(tmp_path, survey_root):
    from akshara.web.app import create_app

    app = create_app(data_root=survey_root, runs_root=tmp_path / 'runs', cache_dir=tmp_path / 'cache')
    with TestClient(app) as client:
        result = client.post('/api/run', json={'path': 'ts_legislature.pdf', 'mode': 'fake', 'max_pages': 1})
        assert result.status_code == 200, result.text
        report = client.get('/api/documents/ts_legislature/report').json()
        assert report['document'] == 'ts_legislature.pdf' and report['complete']
        assert report['pages'][0]['route'] == 'unicode_ok' and report['pages'][0]['spans']
        assert client.get('/api/documents/ts_legislature/pages/1/image').status_code == 200
        answer = client.post('/api/ask', json={'document': 'ts_legislature', 'question': 'What is the budget?', 'mode': 'fake'})
        assert answer.status_code == 200 and answer.json()['abstained']
        assert answer.json()['simulated'] and not answer.json()['citations']


def test_page_images_stay_readable_during_a_question(web, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from importlib import import_module
    from threading import Event

    client, _, _ = web
    client.post('/api/run', json={'path': 'demo/notice.pdf', 'mode': 'fake'})
    module = import_module('akshara.web.app')
    original = module.ask
    started, release = Event(), Event()
    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'ask', blocked)
    with ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, '/api/ask', json={'document': 'notice', 'question': 'What?'})
        try:
            assert started.wait(5)
            assert client.get('/api/documents/notice/report').status_code == 200
            assert client.get('/api/documents/notice/pages/1/image').status_code == 200
        finally:
            release.set()
        assert pending.result().status_code == 200
