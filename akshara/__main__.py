"""CLI for offline triage, survey agreement, and one-document recovery."""

import argparse
import json
from pathlib import Path

from openai import APIError

from .meta_client import FakeClient, MetaClient, MissingKey


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m akshara", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    triage = commands.add_parser("triage", help="Print per-page and document route labels")
    triage.add_argument("pdf", nargs="+", type=Path)
    check = commands.add_parser("check", help="Check triage agreement against the survey CSV")
    check.add_argument("csv", nargs="?", default=Path("data/m0b_results.csv"), type=Path)
    check.add_argument("--pdf-root", default=Path("../telugu-m0"), type=Path)
    execute = commands.add_parser("run", help="Recover, independently verify, and extract one PDF")
    execute.add_argument("pdf", type=Path)
    modes = execute.add_mutually_exclusive_group()
    modes.add_argument("--replay", action="store_true", help="Cache only; fail on any cache miss")
    modes.add_argument("--fake", action="store_true", help="Offline model abstentions; real native/glyph recovery")
    execute.add_argument("--max-pages", type=int)
    execute.add_argument("--cap-usd", type=str, default="3")
    execute.add_argument("--cache-dir", type=Path, default=Path("cache"))
    smoke = commands.add_parser("smoke", help="Three tiny live calls; skip with exit 1 without a key")
    smoke.add_argument("--pdf", type=Path, default=Path("data/demo/ap_2026_I-1.pdf"))
    smoke.add_argument("--page", type=int, default=2)
    evaluate = commands.add_parser("eval", help="Evaluate selected pages of every demo PDF")
    evaluate.add_argument("--mode", choices=("fake", "replay", "live"), default="fake")
    evaluate.add_argument("--docs", type=Path, default=Path("data/demo"))
    evaluate.add_argument("--max-pages", type=int, default=2)
    evaluate.add_argument("--cap-usd", type=str, default="3")
    evaluate.add_argument("--cache-dir", type=Path, default=Path("cache"))
    evaluate.add_argument("--labels", type=Path, default=Path("eval/labels.jsonl"))
    serve = commands.add_parser("serve", help="Start the local web demo")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            client = MetaClient(cap="0.05", max_attempts=1, use_cache=False)
            client._key()  # Before accessing PDFs or constructing an SDK client.
            from .smoke import smoke
            if args.page <= 0:
                parser.error("--page must be positive")
            return 0 if smoke(client, args.pdf, page=args.page) else 1
        elif args.command == "eval":
            from .eval import evaluate
            if args.max_pages <= 0:
                parser.error("--max-pages must be positive")
            path, result = evaluate(args.docs, mode=args.mode, max_pages=args.max_pages,
                cap=args.cap_usd, cache_dir=args.cache_dir, labels=args.labels)
            print(path)
            print(f"{len(result['documents'])} documents; mode={args.mode}; "
                  f"cost_usd={result['usage']['cost_usd']:.8f}")
            return 0 if all(d["complete"] for d in result["documents"]) else 1
        elif args.command == "triage":
            from .triage import triage_pdf
            for pdf in args.pdf:
                result = triage_pdf(pdf)
                print(pdf.name)
                for page in result["pages"]:
                    print(f"  page {page['page']}: {page['label']}")
                print(f"  document: {result['label']}")
        elif args.command == "serve":
            if not 1 <= args.port <= 65535:
                parser.error("--port must be between 1 and 65535")
            import uvicorn
            uvicorn.run("akshara.web.app:app", host="127.0.0.1", port=args.port)
        elif args.command == "check":
            from .triage import check_survey
            result = check_survey(args.csv, args.pdf_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["errors"]:
                return 1
        else:
            if args.max_pages is not None and args.max_pages <= 0:
                parser.error("--max-pages must be positive")
            client_class = FakeClient if args.fake else MetaClient
            client = client_class(cache_dir=args.cache_dir, replay=args.replay,
                                  cap=args.cap_usd)
            if not args.fake and not args.replay:
                client._key()
            from .pipeline import run
            path = run(args.pdf, client, max_pages=args.max_pages)
            print(path)
            print(json.dumps(client.summary()))
            if args.fake:
                print("SIMULATED model reads; native/glyph recovery is real; verification is UNCERTAIN.")
    except MissingKey as exc:
        prefix = "Smoke skipped: " if args.command == "smoke" else ""
        parser.exit(1, prefix + str(exc) + "\n")
    except APIError as exc:
        # SDK error bodies can echo credentials/request content. Never print them.
        parser.exit(1, f"Meta API request failed: {type(exc).__name__}\n")
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
