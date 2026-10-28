"""CLI for offline triage, survey agreement, and one-document recovery."""

import argparse
import json
from pathlib import Path

from .meta_client import CALL_ESTIMATE_USD, FakeClient, MetaClient
from .pipeline import run
from .triage import check_survey, triage_pdf


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
    execute.add_argument("--cap-usd", type=str, default="2")
    execute.add_argument("--call-estimate-usd", type=str, default=CALL_ESTIMATE_USD)
    execute.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = parser.parse_args(argv)
    try:
        if args.command == "triage":
            for pdf in args.pdf:
                result = triage_pdf(pdf)
                print(pdf.name)
                for page in result["pages"]:
                    print(f"  page {page['page']}: {page['label']}")
                print(f"  document: {result['label']}")
        elif args.command == "check":
            result = check_survey(args.csv, args.pdf_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result["errors"]:
                return 1
        else:
            if args.max_pages is not None and args.max_pages <= 0:
                parser.error("--max-pages must be positive")
            client_class = FakeClient if args.fake else MetaClient
            client = client_class(cache_dir=args.cache_dir, replay=args.replay,
                                  cap=args.cap_usd, estimate=args.call_estimate_usd)
            path = run(args.pdf, client, max_pages=args.max_pages)
            print(path)
            print(json.dumps(client.summary()))
            if args.fake:
                print("SIMULATED model reads; native/glyph recovery is real; verification is UNCERTAIN.")
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
