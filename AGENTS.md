# Akshara: agent notes

Read docs/SPEC.md first. Build/test: `.venv/bin/python -m pytest -q`. Run: `.venv/bin/python -m akshara --help`.
Rules: no new dependencies; no network calls in tests (fake client / --replay); MODEL_API_KEY only from the
environment; never commit data/, cache/, runs/, PDFs, secrets, personal data or absolute local paths. Small,
readable modules; no speculative features.
