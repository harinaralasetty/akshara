# Akshara

Akshara recovers Telugu text from PDF text layers, independently reads page images
with Meta models, and extracts facts with page/region evidence. The local web demo
lets you inspect those regions and ask questions whose answers quote the evidence.
Python 3.13; use the existing `.venv`. No added dependencies.

```sh
# Download only the twelve public, non-personal demo documents.
# Run this outside an offline sandbox.
.venv/bin/python scripts/fetch_demo_docs.py

# Offline alternative, if the optional local survey corpus is installed:
.venv/bin/python scripts/fetch_demo_docs.py --from-local

.venv/bin/python -m akshara triage data/demo/ts_legislature.pdf
.venv/bin/python -m akshara run data/demo/ts_legislature.pdf --max-pages 2 --fake
.venv/bin/python -m akshara serve --port 8000
# Open http://127.0.0.1:8000

.venv/bin/python -m pytest -q
AKSHARA_SKIP_SURVEY=1 .venv/bin/python -m pytest -q
```

The viewer lists documents in `runs/`. Pick one, click a page, and click a span,
SAM region, record citation, or answer citation to highlight its source. Page
badges show the recovery route and VERIFIED / CORRECTED / UNCERTAIN status.
Process a PDF by entering its path under `data/`, choosing fake, replay, or live,
and setting the maximum pages. The default is two pages. The server binds to
localhost and accepts existing PDFs under `data/`; there is no upload step.

Fake mode needs no API key or network. It uses real native/glyph recovery and
marks model work as simulated. Default fake reads abstain and produce no invented
records or answers. Scripted `FakeClient` tests exercise successful extraction and
cited answers. Replay needs an exact recorded request. Live makes paid calls and
reads `MODEL_API_KEY` only from the server environment. There are no external
scripts, styles, fonts, or build steps in the viewer.

## Demo documents

The fixed allowlist contains budgets, public laws/notices, a blank form, and an
archive document. The fetcher reads `file`, `url`, and `sha256` from
`data/m0b_results.csv` when present. A checked-in, sanitized twelve-row fallback
lives in `scripts/demo_docs.csv`, so a public checkout does not need the private
survey or its CSV. `--manifest` can select either CSV explicitly. Downloads and
local copies are checksum verified before an atomic replacement; valid existing
files are reused. PDFs remain ignored in `data/demo/`.

`--from-local` copies only these same twelve files from the optional sibling
survey corpus. `--source-root` and `--output-dir` can override those locations.
The script prints actual triage results, never labels copied from the CSV.

| Demo PDF | Actual document route |
| --- | --- |
| ap_2026_I-1.pdf | unicode_ok |
| ap_2026_v5.pdf | legacy_font |
| ap_annex_25-26.pdf | legacy_font |
| ap_gazette_2014.pdf | scanned |
| mcr_rti.pdf | legacy_font |
| neredcherla_rti.pdf | scanned |
| court_telugu_2024.pdf | legacy_font |
| ecourts_telugu_2021.pdf | unicode_ok |
| law_telugu_1956.pdf | scanned |
| ceots_form6.pdf | unclear |
| ts_legislature.pdf | unicode_ok |
| archive_dathumala.pdf | unclear |

This allowlist covers legacy, Unicode, scanned and unclear documents. It contains
**no naturally glyph-ID-routed document** under the current triage rules. The
glyph regression test deliberately removes ToUnicode from a temporary copy of a
public demo document; that modified file is never published as an original demo.
The AP Finance URLs contain literal `...Bud@et` path segments in the source CSV.
They are preserved verbatim; online source availability has not been checked in
the sandbox. Local copying verifies the same expected bytes.

## Document questions

`akshara.agent.ask(report_path, question, client, max_steps=6)` reads one saved
report and its page PNGs. It uses Muse Spark's Chat Completions function tools,
preserving assistant tool calls and matching tool-result IDs as in the Meta
cookbook. All calls go through the existing client cache and spend guard.

- `search_records(query)` finds records by literal substring and returns quotes.
- `recover_page(page)` returns the saved recovered spans.
- `read_region(page, bbox)` crops the saved PNG and asks Spark to read it. New
  reads are UNCERTAIN until independently verified.
- `segment(page, concept)` asks SAM 3.1 for regions. Geometry alone cannot support
  a textual answer; the agent must read the region or recover its page.

The default limit is six model turns, with at most four tool calls per turn.
The final response selects literal quotes from evidence actually returned by
these tools. The application builds the answer from those validated quotes and
derives page, bbox, source and verification from the evidence. Free-form prose,
partial number tokens, invented quotes, unseen citations and exhausted limits produce "I cannot verify
an answer". These are conservative extractive answers; they do not translate,
calculate or synthesize unsupported claims. Simulated evidence stays UNCERTAIN.

The local HTTP endpoints are:

| Method | Path | Payload / result |
| --- | --- | --- |
| GET | `/api/documents` | List processed documents |
| GET | `/api/documents/{document}/report` | Saved report |
| GET | `/api/documents/{document}/pages/{page}/image` | Saved page PNG |
| POST | `/api/run` | `{ "path": "demo/ts_legislature.pdf", "mode": "fake", "max_pages": 2 }` |
| POST | `/api/ask` | `{ "document": "ts_legislature", "question": "What amounts are listed?", "mode": "fake", "max_steps": 6 }` |

Run and ask requests default to fake mode. Replay uses cached real responses;
missing cache entries return HTTP 409. Spend-limit errors also return 409. Live
requires an environment key. A run returns its
document ID and report; ask returns `answer`, `citations`, `abstained`, `steps`,
and `simulated`. PDF, report and image paths are restricted to their roots,
including symlinks. Reports include a content revision; the viewer supplies it
with image and ask requests so a rerun cannot silently invalidate citations.
The local demo processes one run or question at a time.

## Recovery and verification

A run writes `runs/<pdf-stem>/report.json`, `page-NNN.png`, and SAM region crops
for scan pages. Repeating a run replaces the report and its referenced images.
Only images referenced in the current report belong to that run. `complete`
means all **selected** pages finished; `truncated` identifies `--max-pages` runs.
Failures retain a report with `complete: false` and the exception type.

Triage uses font resources, embedded cmaps, ToUnicode, text fractions, and
actually displayed images. Per-page routes are `unicode_ok`, `glyph_id`,
`legacy_font`, `scanned`, and `unclear`. Document rules aggregate font evidence
with legacy > glyph > Unicode precedence. Empty vector outlines are unclear;
an image-only page is scanned. The optional survey check reads its
`classification` column only as the expected answer, never as a triage signal.
Legacy labels are routing candidates, not proof of a visually correct mapping.

**Pre-existing dependency:** `legacylipi`, Hari's indic-legacy-pdf project,
supplies glyph-ID recovery and its packaged legacy mappings. This work does not
claim that dependency as new hackathon work. `recover_pdf(..., page_numbers={...})`
returns positioned text; Akshara preserves its inference warnings. Reverse
shaping yields a possible original string, not proof of original spelling.
The installed package's Telugu Priyaanka mapping is a separate research
prototype, not a supported mapping API. Unsupported Telugu legacy fonts use
vision; Shree-Tel is never passed through the Devanagari Shree-Dev mapping.

A Spark verification read receives the image and no recovered text. Pure local
comparison checks numeric tokens, dates, references, currency/amount tokens,
and all word content, a conservative check covering names. Whitespace and
Unicode decimal digit variants are normalized. Full agreement is VERIFIED;
otherwise Wonder reads independently, also without prior text. Agreement
between both image readers replaces recovery with CORRECTED text and retains
the original spans. Other conflicts, empty reads, and illegibility remain
UNCERTAIN. Verification is page-wide and deliberately conservative: spelling,
line order, or token-boundary differences can make the entire page uncertain.
Agreement is evidence, not ground truth; two readers can share an error.

Extraction uses JSON Schema for document type, office, dates, amounts, reference
numbers, and table rows. Every accepted record has a literal quote, supporting
span IDs, a derived page/bbox, and the weakest supporting verification status.
Unsupported quotes/values/IDs are rejected with reasons. An inferred document
type remains UNCERTAIN unless its value also appears in the evidence.

Pages are 1-based; boxes use unrotated, crop-relative PDF points with top-left
origin, matching PyMuPDF text coordinates. Reports include unrotated bounds and
page rotation so the viewer and cropped reads align evidence with the image.
Vision boxes are converted from normalized image coordinates; SAM boxes come
from source-image pixels. Scan pages segment text blocks, tables, stamps and
signatures and save cited region crops. Vision boxes remain model estimates.
The pipeline targets ordinary horizontal text; complex reading order and
unresolved legacy mappings can require manual review.

## Model calls and cost

All live requests use `https://api.meta.ai/v1`. Spark is `muse-spark-1.3`, the
second reader is `muse-wonder-1.0`, and segmentation is `sam-3.1` through the
Responses stream and `meta_sam_parser`, following the local Meta cookbook.

`--replay` uses real cached responses only and fails on any missing request.
It needs no key and never constructs an SDK client. The request hash includes
model, prompt, images, schema, tools/conversation, and operation. `--cache-dir`
defaults to `cache/`. Fake requests use a separate cache namespace and cannot
seed a real replay.

Every attempted paid call reserves a conservative USD 0.25 by default, from a
USD 2 request cap. Configure CLI `--cap-usd` and `--call-estimate-usd`, or change
`CALL_ESTIMATE_USD` in `akshara/meta_client.py`. SDK retries are disabled; failed
attempts keep their reservation. Cache hits and fake reads cost zero. These
are **estimated reservations, not measured provider charges**; the estimate
must be kept conservative for the actual tariff and request sizes. Raw chat
responses, including reported usage, are retained in the ignored cache.

## Offline validation and optional survey

Tests prohibit socket connections and use scripted FakeClient responses or HTTP
mock transports. Generated PDF/image fixtures cover tool conversations, citation
rejection, abstention, rotations, crops, web requests, replay, spend limits and
checksum failure without any external corpus. Tests that inspect real demo PDFs
skip cleanly when `data/demo/` is absent, or when `AKSHARA_SKIP_SURVEY=1` is set.
They do not require the sibling survey folder.

The full local survey check remains optional for users who have that separate
corpus and the ignored CSV:

```sh
.venv/bin/python -m akshara check data/m0b_results.csv --pdf-root ../telugu-m0
```

No PDF fixtures, data, cache, run outputs, secrets or personal data are tracked.
Live API behavior, reading accuracy and online download availability still need
network-enabled evaluation. No external network calls were made during this build.

By Hari Naralasetty · [GitHub](https://github.com/harinaralasetty) ·
[LinkedIn](https://www.linkedin.com/in/harinaralasetty)
