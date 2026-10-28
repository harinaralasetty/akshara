"""FastAPI adapter around the existing pipeline and evidence-only agent."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, field_validator

from ..agent import ask
from ..meta_client import CacheMiss, FakeClient, MetaClient, SpendLimit
from ..pipeline import run

Mode = Literal["fake", "replay", "live"]


class RunRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    mode: Mode = "fake"
    max_pages: int = Field(default=2, ge=1)


class AskRequest(BaseModel):
    document: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    mode: Mode = "fake"
    max_steps: int = Field(default=6, ge=1, le=20)
    revision: str | None = None

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value.strip()


def contained(root, path):
    """Resolve symlinks before permitting a read or write beneath a demo root."""
    root, path = root.resolve(), path.resolve()
    if path == root or not path.is_relative_to(root):
        raise HTTPException(400, "Path must stay inside its document/data directory")
    return path


def create_app(*, data_root="data", runs_root="runs", cache_dir="cache", client_factory=None):
    data_root, runs_root, cache_dir = map(Path, (data_root, runs_root, cache_dir))
    app = FastAPI(title="Akshara", docs_url=None, redoc_url=None)
    # Serialize processing/questions, but allow viewing pages during a question.
    # Runs also hold the artifact lock to protect report/image snapshots.
    work_lock = Lock()
    artifact_lock = Lock()

    @contextmanager
    def work(lock=work_lock):
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another run or question is in progress")
        try:
            yield
        finally:
            lock.release()

    def new_client(mode):
        if client_factory:
            return client_factory(mode)
        if mode == "fake":
            return FakeClient(cache_dir=cache_dir)
        return MetaClient(cache_dir=cache_dir, replay=mode == "replay")

    def report_path(document):
        if document in (".", "..") or Path(document).name != document:
            raise HTTPException(400, "Invalid document name")
        directory = contained(runs_root, runs_root / document)
        path = contained(directory, directory / "report.json")
        if not path.is_file():
            raise HTTPException(404, "Document report not found")
        return path

    def read_report(path):
        try:
            raw = path.read_bytes()
            report = json.loads(raw)
            report["revision"] = hashlib.sha256(raw).hexdigest()
            return report
        except (OSError, ValueError):
            raise HTTPException(409, "Document report is unreadable") from None

    def pdf_path(value):
        path = Path(value)
        if not path.is_absolute():
            if path.parts and path.parts[0] == data_root.name:
                path = Path(*path.parts[1:])
            path = data_root / path
        path = contained(data_root, path)
        if path.suffix.lower() != ".pdf":
            raise HTTPException(400, "Select a PDF under data/")
        if not path.is_file():
            raise HTTPException(404, "PDF not found under data/")
        # Pipeline names outputs after the PDF stem; constrain its write too.
        directory = contained(runs_root, runs_root / path.stem)
        if directory.is_dir():
            for child in directory.iterdir():
                contained(directory, child)
        return path

    @app.exception_handler(CacheMiss)
    @app.exception_handler(SpendLimit)
    async def model_limit(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).parent / "static" / "index.html", media_type="text/html")

    @app.get("/api/documents")
    def documents():
        result = []
        for directory in sorted(runs_root.iterdir()) if runs_root.is_dir() else []:
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                report = read_report(report_path(directory.name))
                result.append(dict(id=directory.name, document=report.get("document", directory.name),
                                   complete=report.get("complete", False), simulated=report.get("simulated", False),
                                   pages=len(report.get("pages", []))))
            except HTTPException:
                continue
        return result

    @app.get("/api/documents/{document}/report")
    def report(document: str):
        with work(artifact_lock):
            return read_report(report_path(document))

    @app.get("/api/documents/{document}/pages/{page}/image")
    def page_image(document: str, page: int, revision: str | None = None):
        with work(artifact_lock):
            path = report_path(document)
            report = read_report(path)
            if revision is not None and revision != report["revision"]:
                raise HTTPException(409, "Document changed; reload its report")
            selected = next((p for p in report["pages"] if p["page"] == page), None)
            if selected is None:
                raise HTTPException(404, "Processed page not found")
            image = contained(path.parent, path.parent / selected["image"])
            if not image.is_file() or image.suffix.lower() != ".png":
                raise HTTPException(404, "Page image not found")
            # Snapshot under the lock so a rerun cannot replace a streamed file.
            return Response(image.read_bytes(), media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/run")
    def execute(body: RunRequest):
        pdf = pdf_path(body.path)
        with work(), work(artifact_lock):
            try:
                client = new_client(body.mode)
                path = run(pdf, client, max_pages=body.max_pages, output_root=runs_root)
                return dict(document=path.parent.name, report=read_report(path))
            except (CacheMiss, SpendLimit):
                raise
            except Exception as exc:
                # Provider errors may include request data or paths. Return only type.
                raise HTTPException(400, f"Processing failed: {type(exc).__name__}") from None

    @app.post("/api/ask")
    def question(body: AskRequest):
        with work():
            path = report_path(body.document)
            if body.revision is not None and body.revision != read_report(path)["revision"]:
                raise HTTPException(409, "Document changed; reload its report")
            try:
                return ask(path, body.question, new_client(body.mode), max_steps=body.max_steps)
            except (CacheMiss, SpendLimit):
                raise
            except Exception as exc:
                raise HTTPException(400, f"Question failed: {type(exc).__name__}") from None

    return app


app = create_app()
