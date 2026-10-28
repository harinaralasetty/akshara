# Akshara hackathon provenance and demo

Akshara is a verifying reading agent for Telugu government PDFs. This repository
started **2026-10-28** during the Meta Global AI Developer Hackathon. The scope
and judging weights below follow the project [build spec](SPEC.md); they are not
a claim that eligibility or scoring has been independently confirmed.

## What was built during the hackathon

New work in this repository includes per-page PDF triage, recovery routing around
an existing dependency, independent image-reader verification and conservative
comparison, structured record extraction with validated quotes and bounding
boxes, and a bounded tool-calling question agent. It also includes the local web
viewer with clickable evidence, fake/replay modes, cache and usage accounting,
retry/spend controls, live smoke probes, offline tests, evaluation harness,
public-demo preparation and labelled synthetic glyph-demo generator.

The web demo opens existing PDFs under `data/`; it does not implement the upload
flow proposed in the spec. Unsupported Telugu legacy mappings fall back to
vision. Fake mode abstains on vision, extraction and question answering, while
native/glyph recovery runs locally. Tests with scripted responses exercise
successful extraction and cited answers; those are not live-model results.

## What predates the hackathon

- **legacylipi / indic-legacy-pdf:** Hari's earlier work on an existing open-source
  project supplies glyph-ID reverse shaping and packaged legacy mappings.
  Akshara integrates it; it does not claim to have built that recovery engine
  during the event. Upstream and fork attribution remains in the dependency.
- **Local survey data:** the earlier Telugu PDF corpus and its classification
  survey exist outside this repository. They motivated the problem and triage
  rules. Neither that survey dataset nor its aggregate findings are a new
  hackathon benchmark. The checked-in public allowlist contains only source URLs
  and checksums; PDFs and the local survey CSV are not tracked.
- **External tooling:** Meta models, the Meta cookbook/parser, SDK, PDF libraries
  and the other installed packages are dependencies, not original contributions.

The public demo has twelve source documents plus one synthetic derivative. The
derivative removes Telugu ToUnicode maps from page 2 of `ap_2026_I-1.pdf`. Its
filename, PDF metadata and web badge say **SYNTHETIC**. It demonstrates a controlled
failure mode; it is not an untouched government PDF or evidence of naturally
occurring glyph-ID coverage in this allowlist.

## Mapping to judging criteria

| Criterion | Weight | What this build demonstrates | Evidence and limits |
| --- | ---: | --- | --- |
| Impact | 30% | Makes difficult Telugu public documents inspectable, with page evidence and explicit uncertainty. | Public budgets, laws, notices and a blank form are demo inputs. No measured user adoption, time saved, or human reading-accuracy result is claimed. |
| Use of Meta tooling | 25% | Muse Spark recovery, independent image reads, structured extraction and tool calls; Muse Wonder disagreement read; SAM 3.1 region segmentation. | Request shapes, streams, usage and retry behavior are tested offline. Live Meta API behavior remains unverified until an environment key is used successfully. Fake mode demonstrates application flow and abstention. |
| Technical depth | 20% | Per-page routes, independent verification, evidence-derived citations, crop/rotation handling, cache isolation, spend accounting and an evaluation harness. | Tests cover supported behavior and failure paths. Agreement is not truth; reverse shaping has ambiguity and vision boxes are estimates. The recovery engine is pre-existing. |
| Demo | 25% | Local web workflow from a labelled PDF through recovered text, highlighted evidence and an abstaining question. | The script below is a reproducible offline demonstration. Successful live OCR, SAM regions, extracted records and answers must only be shown after a real validated run. |

## Three-minute video: preparation

Use only the curated public demo files. With those files already prepared locally:

```sh
.venv/bin/python scripts/make_glyph_demo.py
.venv/bin/python -m akshara eval --mode fake --docs data/demo
.venv/bin/python -m akshara serve --port 8000
```

If the optional survey corpus is present, the offline corpus preparation command
is `.venv/bin/python scripts/fetch_demo_docs.py --from-local --glyph-demo`.
The public checkout can fetch the same allowlist in a network-enabled setup;
online source availability has not been verified in this build.

Open the web demo at localhost port 8000. Keep the synthetic glyph document,
`ap_2026_I-1`, and `law_telugu_1956` available in the document selector. Use a
large browser window and readable zoom. Frame out terminals containing keys,
home-directory paths and unrelated files. Keep fake mode and its simulated
notice visible. Do not seed successful fake answers for the recording.

## Three-minute video: timed script and shot list

| Time | Shot / action in the web demo | Narration |
| --- | --- | --- |
| 0:00–0:20 | Start on the viewer with the synthetic document selected. Show the title, rendered Telugu page and SYNTHETIC / simulated notices. | “Akshara helps read Telugu public PDFs when their text layer is broken. A page can look readable while copy and search fail. It keeps the source page beside recovered text and says when it cannot verify an answer.” |
| 0:20–0:40 | Point to the document selector and the SYNTHETIC badge, then the glyph_id page badge. | “This is a controlled example: I removed the Telugu Unicode mapping from a public budget page. The filename, PDF metadata and this badge label it synthetic. It demonstrates glyph-ID recovery without passing a modified document off as an original.” |
| 0:40–1:15 | Click a recovered Telugu span overlay. Hold on its highlighted box and the evidence panel; click a second span. | “The recovered Unicode text comes from legacylipi, a pre-existing dependency. Akshara's new work routes the page, preserves its position and exposes the evidence here. Selecting a span shows its text and page region. Recovery can be ambiguous, so this offline run keeps the result uncertain.” |
| 1:15–1:45 | Select `ap_2026_I-1`; switch from page 1 to page 2. Show the route badges and select a native-text region. | “Routing happens per page. A scanned cover and a native-text contents page can take different paths in the same PDF. In a live run, Muse Spark independently reads the image; a disagreement triggers Muse Wonder. Agreement can verify or correct a reading. The labels describe reader agreement, not guaranteed truth.” |
| 1:45–2:10 | Select `law_telugu_1956`, page 1. Show the scanned badge, empty records and simulated notice. | “Scan pages need model reading. The implemented live path also asks SAM 3.1 for text, table, stamp and signature regions. This is fake mode: it makes no paid calls and invents no text or records. Live Meta behavior is still unverified until I use a key.” |
| 2:10–2:35 | Return to the synthetic document. Ask “What financial years are listed?” in fake mode. Hold on the abstention and empty citation area. | “Questions use bounded tools to search records, recover saved pages, read crops and segment regions. Answers must quote returned evidence with a page and box. Here the fake model abstains. A successful answer and its citation should only be shown from a validated run.” |
| 2:35–3:00 | Hold on the highlighted page and uncertainty badge. End with the visible author footer; an on-screen caption can show `python -m akshara eval --mode fake`. | “This repository began on October 28, 2026. The new work is the verifying agent, evidence pipeline, web viewer and evaluation. The recovery dependency and earlier local survey predate the event. The report measures route coverage, text, verification and usage cost; human-labelled accuracy is still pending.” |

Total: **180 seconds**. The script deliberately remains valid without a model
key. If recording a verified live or real replay run later, identify that mode
and the actual observed results. Never narrate a fake badge as independent model
verification. Keep synthetic recovery and naturally occurring document results
distinct in any submission metrics.

## Remaining validation

Run the three live smoke probes only in an explicitly network-enabled session
with `MODEL_API_KEY` exported: `python -m akshara smoke`. Then inspect real model
reads, regions, records and usage before claiming live integration. A Telugu
reader must fill the blank label template and inspect the source images before
field accuracy can be reported. See [evaluation semantics](EVALUATION.md).
