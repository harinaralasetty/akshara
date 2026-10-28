# Evaluation

Run from the repository root with the existing Python 3.13 environment:

```sh
.venv/bin/python -m akshara eval --mode fake --docs data/demo
```

The default selects the first two PDF pages of every `.pdf` in that directory.
Use `--max-pages N` to change the selection. A document is complete when all
selected pages finish, even if the document has more pages. The report records
the full document length and the selected-page limit; it is not a whole-corpus
reading-accuracy benchmark.

Outputs are `runs/eval/report.md`, `runs/eval/metrics.json`, and the usual
`runs/<pdf-stem>/report.json` files with page images. Rerunning replaces these
outputs. PDF stems must be unique within an evaluation directory.

## Modes and cost

- `fake` uses real native/glyph recovery. Model reads abstain, so unsupported
  legacy, scan and unclear pages normally have no text. All verification stays
  UNCERTAIN. Fake metrics measure offline plumbing and local recovery coverage.
- `replay` uses exact recorded real requests in `cache/`, without a key or SDK
  connection. A miss leaves an incomplete document report and evaluation
  continues with the other documents. Fake caches cannot satisfy real replay.
- `live` requires `MODEL_API_KEY` in the process environment. It uses paid Meta
  calls with a shared `--cap-usd` budget across the entire evaluation, including
  labelled questions. Live provider behavior has not been verified during this
  offline build. Do not treat fake/replay tests as a successful live run.

`--cache-dir` selects the cache. The default cap is USD 3. Each attempt reserves
a request-size estimate, then reconciles it against reported usage and the
tariffs in `akshara/meta_client.py`. Failed or unreported calls retain unresolved
reservations. Confirm tariffs before a paid run; these are application-side
calculations, not an invoice or a hard provider spending limit.

Per-document and per-route tables count completed pages, Unicode Telugu code
points, pages with any nonblank recovered text and their share, page verification
statuses, and measured USD per completed page. “Pages with text” includes any
script; Telugu count is separate. A Unicode route can still contain broken
characters. Neither count establishes reading accuracy.

Cache replay contributes recorded token counts but zero new cost. Usage totals
also include failed page attempts and labelled question calls, which are outside
completed-page cost denominators. Unresolved USD is shown separately and must
not be interpreted as zero cost. Verification means conservative reader
agreement, not ground truth. The SYNTHETIC glyph document is identified separately
and does not establish performance on naturally occurring glyph-ID PDFs.

## Human labels

`eval/labels.template.jsonl` has 18 questions on the twelve public source demo
documents. Every answer is deliberately empty. The synthetic derivative is
excluded from this annotation set.

```sh
cp eval/labels.template.jsonl eval/labels.jsonl
# A Telugu reader fills answer strings after inspecting the actual page images.
.venv/bin/python -m akshara eval --mode fake --labels eval/labels.jsonl
```

Each row has `document` (PDF basename), `field` (grouping label), `question`,
`answer`, and an advisory 1-based `page`. The page is included in each question;
it does not change which pages the pipeline selects. Keep targets within the
selected pages, or increase `--max-pages`. Blank answers are ignored. Leave a
question blank when its premise is absent or unreadable; do not infer an answer
from the filename or the template wording. Use exact visible quotes, preserving
punctuation and digits, rather than translations or explanatory sentences.

Field accuracy here is **exact match of the evidence-only agent's quoted answer**,
grouped by field and document. It does not separately grade every extracted record.
NFC and whitespace are normalized. It gives no partial or semantic credit and
does not normalize punctuation, translate, or equate differing numeric formats.
Abstentions and model/operational errors count as incorrect for evaluated
documents. Labels naming documents outside the evaluation are reported as
unscored. With no filled labels, the result is “Not evaluated”, never 0% accuracy.

Fake model abstentions are expected even with filled labels. Human annotations
and a verified live or recorded run are still needed for a useful accuracy score.
Keep annotations limited to public, non-personal document content.
