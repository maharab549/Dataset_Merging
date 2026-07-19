import argparse
import json
import logging
import sys

from .merge import run


def main():
    parser = argparse.ArgumentParser(prog="weave", description="Combine multiple Hugging Face datasets into one.")
    parser.add_argument("--token", default=None, help="HF token (else reads HF_TOKEN env var). Only needed for gated/private datasets or push-to-hub.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Load each dataset and report detected format/columns without merging. Run this first.")
    p_inspect.add_argument("config")

    p_merge = sub.add_parser("merge", help="Run the full pipeline and write the merged dataset.")
    p_merge.add_argument("config")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.command == "inspect":
        summary = run(args.config, hf_token=args.token, inspect_only=True)
        print(json.dumps(summary, indent=2, default=str))
    elif args.command == "merge":
        summary = run(args.config, hf_token=args.token, inspect_only=False)
        print("\n=== Merge summary ===")
        for s in summary["sources"]:
            print(f"  {s['repo_id']:<45} format={s['format']:<16} rows={s['rows']}")
        print(f"\nRows before filters:  {summary['rows_before_filters']}")
        print(f"Rows after filters:   {summary['rows_after_filters']}")
        print(f"Duplicates removed:   {summary['duplicates_removed']}")
        print(f"Final rows:           {summary['final_rows']}")
        print(f"Splits:               {summary['splits']}")
        print(f"Saved to:             {summary['save_path']}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
