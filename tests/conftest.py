"""Offline tests: fail immediately if any test attempts network I/O."""

import os
import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


@pytest.fixture(scope="session")
def survey_root():
    if os.environ.get("AKSHARA_SKIP_SURVEY") == "1":
        pytest.skip("Optional public demo survey disabled by AKSHARA_SKIP_SURVEY")
    root = Path(__file__).resolve().parents[1] / "data/demo"
    required = ("ap_2026_I-1.pdf", "ap_2026_v5.pdf", "ap_annex_25-26.pdf",
                "ap_gazette_2014.pdf", "mcr_rti.pdf", "neredcherla_rti.pdf",
                "court_telugu_2024.pdf", "ecourts_telugu_2021.pdf", "law_telugu_1956.pdf",
                "ceots_form6.pdf", "ts_legislature.pdf", "archive_dathumala.pdf")
    if not all((root / name).is_file() for name in required):
        pytest.skip("Public demo documents are absent; run scripts/fetch_demo_docs.py")
    return root
