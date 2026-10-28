# Akshara: build spec (Meta Global AI Developer Hackathon, Oct 27 - Nov 4 2026)

## One line
Akshara is a verifying reading agent for Telugu government PDFs that ordinary tools can't read (broken legacy
fonts, glyph-ID text, scans). It recovers the text, checks it against the page image with Meta's models, extracts
structured facts with a citation to the exact page region, and says "not sure" instead of guessing.

Not a "basic RAG app" (that category is discouraged by the hackathon rules): the core is per-page recovery
routing, multimodal cross-verification and region-grounded, evidence-backed extraction.

## Problem (firsthand)
Many Telugu government notices, budgets, gazettes and forms are PDFs whose text layer is broken: copy/paste gives
Latin garbage (legacy 8-bit fonts) or raw glyph IDs (embedded fonts without a ToUnicode map), or the page is a
scan. Search, translation and chatbots fail on them. A survey of 65 public Telugu PDFs (local data, see Data)
found 10 glyph-ID, 20 legacy-font, 17 Unicode, 11 scanned and 7 unclear documents.

## Hackathon rules that constrain the build
- Meta models are required: Meta Model API (Muse Spark, Muse Wonder, Muse Image, Muse Voice Transcribe, SAM 3.1)
  and/or open-weight models. Judging: Impact 30%, creative use of Meta tooling 25%, technical depth 20%, demo 25%.
- Only work built during the event counts and the demo must make clear what is new. This repo is new (started
  2026-10-28). `legacylipi` (indic-legacy-pdf, Hari's earlier open-source project) is used as a declared,
  pre-existing dependency; the README must say so.
- Public repo; 3-minute demo video. No personal data in the repo.

## Meta Model API facts (from the official cookbook, local copy ../hackathons/meta/cookbook)
- Base URL `https://api.meta.ai/v1`, `Authorization: Bearer $MODEL_API_KEY`; OpenAI SDK with `base_url` works
  (chat completions, tool calling, structured output via `response_format={"type":"json_schema",...}`, images as
  `image_url` data URLs).
- Model ids: `muse-spark-1.3` (text/image/PDF in, text out, tool calling, structured output), `muse-wonder-1.0`
  (preview, reasoning effort minimal..max), `sam-3.1` (text prompt + image -> boxes and masks; see
  cookbook/07_sam/01_api_basics/python for the exact request and the `meta_sam_parser` package).
- $150 credit budget in total: every call goes through a disk cache and a spend guard.

## Architecture (package `akshara/`)
1. triage: per-page signal (unicode_ok / glyph_id / legacy_font / scanned / unclear) from PyMuPDF font and text
   statistics (port the essence of telugu-m0's rules; new code, small).
2. recover: route per page. unicode_ok -> native text. glyph_id -> legacylipi glyph-ID recovery. legacy_font ->
   legacylipi mapping when the font is supported, else vision. scanned/unclear -> Muse Spark page-image reading.
   Every span keeps its page number and bounding box.
3. verify: Muse Spark reads the rendered page image independently; spans compared field by field (numbers,
   dates, names, GO numbers, amounts). Agreement -> VERIFIED; disagreement -> second independent read
   (Muse Wonder) decides CORRECTED or UNCERTAIN. Scans: SAM 3.1 segments regions (table, stamp, signature,
   text block) so citations point to a region crop.
4. extract: structured records (document type, issuing office, dates, amounts, reference numbers, table rows)
   via structured output, each with page + bbox + verification status.
5. agent: tool-calling question answering over one document (tools: search_records, read_region(page,bbox),
   recover_page(page), segment(page, concept)); answers cite regions or abstain.
6. web: FastAPI + one static page: upload PDF, per-page route and verification badges, records table, click a
   citation to see the page with the region highlighted, ask box.
7. eval: per-category harness on the local survey set; metrics: Telugu recovery coverage, verifier agreement,
   field accuracy on a small hand-labelled set (labels by Hari later), cost per page.

## Engineering rules
- Python 3.13, venv at `.venv` (already created; no network inside the build sandbox: do not add new
  dependencies beyond those installed: openai, meta_sam_parser, pymupdf, pypdfium2, pillow, fastapi, uvicorn,
  python-multipart, pytest, fonttools, uharfbuzz, click, rich, pyyaml, httpx, tenacity, legacylipi (editable)).
- Meta client: one module; `MODEL_API_KEY` from env only; disk cache keyed by request hash; spend guard with a
  per-run cap; `--replay` mode that serves recorded responses so tests and the demo run without a key or cost.
- Tests run offline with a fake client. No secrets, no personal data, no local absolute paths in the repo.
- Data: survey PDFs live outside the repo (../telugu-m0) and in `data/` which is git-ignored. Demo documents must
  be public, non-personal (budgets, acts, blank forms, public notices).
