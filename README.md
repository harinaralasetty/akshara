# Akshara — round 1

Akshara recovers Telugu text from PDF text layers, independently reads page images
with Meta models, and extracts structured facts with page/region evidence.
Python 3.13; use the existing `.venv`. No added dependencies.

```sh
.venv/bin/python -m akshara --help
.venv/bin/python -m akshara triage ../telugu-m0/pdfs/vsp_1.pdf
.venv/bin/python -m akshara check data/m0b_results.csv --pdf-root ../telugu-m0
.venv/bin/python -m akshara run ../telugu-m0/pdfs/vsp_1.pdf --max-pages 2 --fake
.venv/bin/python -m pytest -q
```

A run writes `runs/<pdf-stem>/report.json`, `page-NNN.png`, and SAM region crops
for scan pages. Repeating a run replaces the report and its referenced images.
Only images referenced in the current report belong to that run. `complete`
means all **selected** pages finished; `truncated` identifies `--max-pages` runs.
Failures retain a report with `complete: false` and the exception type.

For live reads, set `MODEL_API_KEY` in the environment and omit `--fake`.
All requests use `https://api.meta.ai/v1`. Spark is `muse-spark-1.3`, the second
reader is `muse-wonder-1.0`, and segmentation is `sam-3.1` through the Responses
stream and `meta_sam_parser`, following the local Meta cookbook.

`--replay` uses real cached responses only and fails on any missing request.
It needs no key and never constructs an SDK client. The request hash includes
model, prompt, images, schema, and operation. `--cache-dir` defaults to `cache/`.
`--fake` uses a separate cache namespace and cannot seed a real replay. Its
default model reads abstain; native and legacylipi recovery still run. Reports
are explicitly simulated, with UNCERTAIN verification and no invented records.
Tests can supply scripted FakeClient responses or a request handler.

Every attempted paid call reserves a conservative USD 0.25 by default, from a
USD 2 run cap. Configure `--cap-usd` and `--call-estimate-usd`, or change
`CALL_ESTIMATE_USD` in `akshara/meta_client.py`. SDK retries are disabled; failed
attempts keep their reservation. Cache hits and fake reads cost zero. These
are **estimated reservations, not measured provider charges**; the estimate
must be kept conservative for the actual tariff and request sizes. Raw chat
responses, including reported usage, are retained in the ignored cache.

## Recovery and verification

Triage uses font resources, embedded cmaps, ToUnicode, text fractions, and
actually displayed images. Per-page routes are `unicode_ok`, `glyph_id`,
`legacy_font`, `scanned`, and `unclear`. Document rules aggregate font evidence
with legacy > glyph > Unicode precedence. Empty vector outlines are unclear;
an image-only page is scanned. The survey check reads its `classification`
column only as the expected answer. It does not use CSV signals to classify.
Legacy labels are routing candidates, not proof of a visually correct mapping.

**Pre-existing dependency:** `legacylipi` (Hari's indic-legacy-pdf project) supplies
glyph-ID recovery and its packaged legacy mappings. This work does not claim
that dependency as new hackathon work. `recover_pdf(..., page_numbers={...})`
returns positioned text; Akshara preserves its inference warnings. Reverse
shaping yields a possible original string, not proof of original spelling.
The installed package's Telugu Priyaanka mapping is a separate research
prototype, not a supported mapping API. Unsupported Telugu legacy fonts use
vision; Shree-Tel is never passed through the Devanagari Shree-Dev mapping.

A Spark verification read receives the image and no recovered text. Pure local
comparison checks numeric tokens, dates, references, currency/amount tokens,
and all word content (a conservative check covering names). Whitespace and
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
origin, matching PyMuPDF text coordinates. Vision boxes are converted from
normalized image coordinates, including page rotation; SAM boxes come from
source-image pixels. Scan pages segment text blocks, tables, stamps, and
signatures and save cited region crops. Vision boxes remain model estimates.
The pipeline currently targets ordinary horizontal text; complex reading order
and unresolved legacy mappings can require manual review.

## Offline validation

Tests prohibit socket connections, use FakeClient or HTTP mock transports, and
exercise all five survey categories, native/glyph end-to-end recovery, structured
grounding, independent verification, cache/replay, spend limits, and streamed
SAM parsing. Survey tests use the read-only sibling `../telugu-m0` corpus and
skip when it is absent. No PDF fixtures, data, cache, run outputs, or secrets are
tracked. Live API behavior and reading accuracy require later network-enabled
evaluation; no network calls were made during this build.
