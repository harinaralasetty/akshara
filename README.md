<div align="center">

# Akshara

### Read difficult Telugu PDFs with page evidence and explicit uncertainty.

**A verifying reading agent for Telugu government PDFs.** Legacy fonts, missing Unicode maps, scanned pages, structured facts and document questions, with evidence you can inspect on the page.

[**Quick start**](#quick-start) · [Why Akshara](#why-akshara) · [What it does](#what-it-does) · [How it works](#how-it-works) · [FAQ](#faq) · [Docs](#documentation) · [MIT license](LICENSE)

```sh
uv pip install --python .venv -r requirements.txt
```

Python 3.13 and uv; glyph-engine setup is below. Use the existing environment for offline work.

</div>

> [!NOTE]
> Live Meta API behavior is **unverified until a key is used successfully**. Offline tests check request shapes and application behavior; fake model reads abstain. Recovery coverage and reader agreement do not establish reading accuracy. Human-labelled field accuracy is still pending.

## Why Akshara

A Telugu PDF can look readable while copying, searching or feeding it to a chatbot produces broken text.

- **Route each page.** Native text, glyph IDs, legacy fonts and scans need different recovery paths, even within one document.
- **Inspect the source.** Recovered spans, accepted records and answers retain page numbers and bounding boxes you can highlight in the web viewer.
- **Keep uncertainty visible.** Independent image reads can verify or correct recovery; unresolved disagreement stays UNCERTAIN. Unsupported answers abstain.
- **Rehearse offline.** Fake mode runs local recovery without a key. Replay uses recorded real requests without making new API calls.

## Quick start

Run commands from this repository's root. The hackathon environment already has `.venv` and the required packages; **skip installation there**. For a fresh checkout, use Python 3.13 and uv:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv -r requirements.txt
# Install the pre-existing glyph engine at the revision used by this build.
# --no-deps avoids its unused translation/OCR application dependencies.
uv pip install --python .venv --no-deps "legacylipi @ git+https://github.com/harinaralasetty/indic-legacy-pdf.git@6043fd5ed7a4448bbe94c77e7cf36ff096d07334"
```

These installation commands require network access or populated local package caches. No packages were installed during the offline build. [requirements.txt](requirements.txt) records the existing environment's package versions; the glyph dependency is installed separately.

**Prepare public demo PDFs**, if the optional local survey corpus is available:

```sh
.venv/bin/python scripts/fetch_demo_docs.py --from-local --glyph-demo
```

If the twelve public PDFs are already in `data/demo/`, create just the derivative:

```sh
.venv/bin/python scripts/make_glyph_demo.py
```

**Use** the CLI and local web demo:

```sh
.venv/bin/python -m akshara triage data/demo/glyph_demo_SYNTHETIC_ap_2026_I-1.pdf
.venv/bin/python -m akshara eval --mode fake --docs data/demo
.venv/bin/python -m akshara serve --port 8000
```

Open `http://127.0.0.1:8000`. Pick a processed document, select a page and click a text region to inspect its evidence. Evaluation prepares the document reports the viewer lists. You can also process an existing PDF under `data/` from the web form. The web demo uses existing local files; there is no upload step.

<details>
<summary><b>Modes, flags, demo preparation and output rules</b></summary>

<br>

- `run PDF --fake --max-pages 2` recovers selected pages into `runs/<pdf-stem>/report.json` with page PNGs. Without `--max-pages`, `run` processes the whole document.
- `eval --mode fake|replay|live --docs data/demo` defaults to fake and the first two pages per document. It writes `runs/eval/report.md` and `metrics.json`. `--labels eval/labels.jsonl` supplies human answers; blank slots are ignored.
- Fake mode keeps native/glyph recovery real, marks model work simulated and leaves verification UNCERTAIN. It invents no records or answers. Scripted tests cover successful extraction and cited answers separately.
- CLI `run --replay` and eval/web replay need exact cached real requests. A miss fails that document; replay never falls back to live. Fake and real cache namespaces are isolated.
- Live `run` (neither `--fake` nor `--replay`), `eval --mode live`, and web live mode read `MODEL_API_KEY` **only from the process environment**. Export it in the shell that starts the CLI/server. Never put keys in the UI, repository or command history.
- `--cap-usd` defaults to 3 for run/eval; eval shares that cap across documents and label questions. `--cache-dir` defaults to `cache/`. Web run/ask requests default to fake and two selected pages for processing.
- Live calls reserve a request-size estimate, then reconcile reported usage against configured model tariffs. Failed or unreported calls keep unresolved reservations. SDK retries are disabled; bounded application retries reserve each attempt. Cache hits and fake calls incur no new charge. This is application-side accounting, not a provider billing limit.
- `python -m akshara smoke` makes three small uncached live probes (text, image, SAM) with no retries. Without an environment key it exits 1 before opening a PDF or constructing an API client. It defaults to a crop of page 2 of `ap_2026_I-1.pdf`.
- In a network-enabled setup, `scripts/fetch_demo_docs.py --glyph-demo` fetches only the twelve public files in the [manifest](scripts/demo_docs.csv). Downloads and local copies are SHA256 checked. The bundled manifest works without the private survey CSV; online source availability has not been verified here.
- `--from-local` uses the optional sibling survey corpus; `--source-root`, `--manifest` and `--output-dir` override its locations. The synthetic derivative is generated separately from one public page and labelled in its filename, metadata and UI.
- Repeating a run replaces its report and referenced images. `complete` covers selected pages; `truncated` records a page limit. Failures preserve an incomplete report. Eval returns nonzero if any document is incomplete.
- Pages are 1-based. Boxes use unrotated, crop-relative PDF points with a top-left origin. The viewer handles page rotation and revisions when highlighting citations.
- Data, PDFs, caches, run outputs and secrets stay untracked. The localhost server accepts paths only under `data/` and serializes processing/questions.

</details>

## What it does

| Area | Behavior |
| --- | --- |
| **Triage** | Per-page `unicode_ok`, `glyph_id`, `legacy_font`, `scanned` or `unclear` routes from text, images and font resources. |
| **Recovery** | Native text or pre-existing legacylipi glyph/mapping recovery; unsupported legacy fonts and scans use image reading in live mode. |
| **Verification** | Muse Spark reads the page without seeing recovered text; disagreement triggers an independent Muse Wonder read. Conservative comparison yields VERIFIED, CORRECTED or UNCERTAIN. |
| **Regions and records** | SAM 3.1 scan regions; structured document types, offices, dates, amounts, references and table rows. Accepted records cite literal supporting text and derived boxes. |
| **Questions** | A bounded tool-calling agent searches records, recovers saved pages, reads crops and segments regions. Answers quote returned evidence with citations or abstain. |
| **Web demo** | Route/status badges, page images, span/region overlays, records and answer citations. Synthetic and simulated results are labelled. |
| **Evaluation** | Per-document and per-route page counts, Telugu characters, text coverage, verification counts and usage cost per page; optional exact-match field questions. |

The public allowlist has twelve source PDFs: budgets, laws/notices, a blank form and an archive document. Adding the synthetic glyph example makes thirteen demo files. Document-level routes can differ from the routes of the first two pages selected by eval. This allowlist has no naturally occurring glyph-ID document under the current rules.

## How it works

Akshara inspects each page's text layer and embedded fonts, then selects a recovery path. Independent image readers check the result without receiving the recovered text. Extraction validates quoted evidence, and the question agent can only return quotes from evidence its tools actually retrieved.

The viewer keeps the page image beside that evidence. Scanned pages can receive SAM regions and saved crops; region geometry alone never supports a textual answer. All model calls pass through one cache, usage ledger and spend guard in [meta_client.py](akshara/meta_client.py).

**Pre-existing dependency:** [legacylipi / indic-legacy-pdf](https://github.com/harinaralasetty/indic-legacy-pdf) supplies glyph-ID reverse shaping and packaged legacy mappings. Akshara integrates that earlier work; it does not claim the recovery engine as a hackathon contribution. The local survey data also predates this repo and is not included. New work here began on **2026-10-28**; see [hackathon provenance and demo script](docs/HACKATHON.md).

Limits matter: reverse shaping can return a possible spelling, reader agreement can share an error, and vision boxes are estimates. The installed dependency has no supported packaged Telugu legacy mapping; those pages use vision. A `unicode_ok` route is a heuristic and may still contain damaged text. The pipeline targets ordinary horizontal text; complex layout needs manual review. More detail is in the [build spec](docs/SPEC.md) and [evaluation guide](docs/EVALUATION.md).

## FAQ

**Can I run it without a Meta key?** Yes: fake mode runs offline and keeps local recovery real. Model work abstains. Real replay also needs no key, but requires previously recorded requests; recordings are not bundled.

**Does fake mode prove the models work?** No. Offline tests validate request/response handling, citations and failure paths. Live Meta behavior and reading accuracy remain unverified until a key is used and the outputs are reviewed.

**Is VERIFIED a guarantee?** No. It means conservative agreement between recovered text and an independent image read. Two readers can share an error. Check the cited source before relying on a fact.

**Does it translate or calculate answers?** The current question agent selects literal quotes from retrieved evidence. It does not synthesize free-form translations or unsupported calculations. Missing support produces “I cannot verify an answer”.

**What is synthetic in the demo?** Only the labelled glyph derivative: Telugu ToUnicode maps are removed from page 2 of a public budget. Its rendered page is retained; the damaged text layer is intentional. Fake model activity is separately labelled simulated.

**What gets sent to Meta?** Live mode sends page images/crops and task text; extraction and question requests include recovered evidence. Fake and replay do not send API requests. Use only public, non-personal demo documents. Responses and usage are cached locally in ignored directories.

**Where are the accuracy numbers?** They are pending human annotation. The [label template](eval/labels.template.jsonl) has 18 questions with empty answers. Eval reports “Not evaluated” until filled labels are available; character coverage is not reading accuracy.

## Documentation

- [Build spec and architecture](docs/SPEC.md)
- [Evaluation modes, metrics and human labels](docs/EVALUATION.md)
- [Hackathon provenance, judging criteria and three-minute demo script](docs/HACKATHON.md)
- [Public demo manifest](scripts/demo_docs.csv) · [Glyph demo generator](scripts/make_glyph_demo.py)
- [Agent instructions](AGENTS.md)

## Contributing

Read [AGENTS.md](AGENTS.md) and [docs/SPEC.md](docs/SPEC.md). Keep modules small, add no dependencies, and use fake clients or replay for tests. Tests block network connections. Do not commit PDFs, data, caches, runs, secrets, personal data or absolute local paths.

```sh
.venv/bin/python -m pytest -q
AKSHARA_SKIP_SURVEY=1 .venv/bin/python -m pytest -q
```

The second command skips optional tests that inspect locally prepared public demo PDFs. Core tests generate their own fixtures and do not need the survey corpus.

## License

[MIT](LICENSE). Dependencies retain their own licenses.

By Hari Naralasetty · [GitHub](https://github.com/harinaralasetty) · [LinkedIn](https://www.linkedin.com/in/harinaralasetty)
