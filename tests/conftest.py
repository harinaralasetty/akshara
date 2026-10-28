"""Offline tests: fail immediately if any test attempts network I/O."""

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
    root = Path(__file__).resolve().parents[2] / "telugu-m0"
    if not root.is_dir():
        pytest.skip("External read-only survey corpus is not installed")
    return root
