"""Run source-grounded semantic equivalence evaluation for v2 graph outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation import (  # noqa: E402
    PipelineArtifacts,
    evaluate_gold_records,
    llm_config_from_env,
    load_gold_jsonl,
    write_gold_template,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether v2 typed graph outputs are source-grounded and semantically equivalent to gold recommendation frames."
    )
    parser.add_argument(
        "--gen-dir",
        required=True,
        help="Pipeline output directory containing terms.json, predicates.json, rules.json, provenances.json, etc.",
    )
    parser.add_argument(
        "--gold",
        help="Gold frame JSONL file. Required unless --write-gold-template is used.",
    )
    parser.add_argument(
        "--out-dir",
        help="Evaluation output directory. Defaults to <gen-dir>/eval_source_grounded.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of gold records to evaluate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM judge temperature.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build evaluation contexts; do not call the LLM judge.",
    )
    parser.add_argument(
        "--write-gold-template",
        help="Write a starter gold JSONL template from provenances.json and exit.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not delete an existing evaluation_results.jsonl before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir) if args.out_dir else gen_dir / "eval_source_grounded"

    if args.write_gold_template:
        count = write_gold_template(gen_dir=gen_dir, output_path=args.write_gold_template, limit=args.max_items)
        print(f"[eval] wrote {count} gold template rows -> {args.write_gold_template}")
        return 0

    if not args.gold:
        raise SystemExit("--gold is required unless --write-gold-template is used.")

    artifacts = PipelineArtifacts.load(gen_dir)
    gold_records = load_gold_jsonl(args.gold)
    llm_config = llm_config_from_env(temperature=args.temperature)

    summary = evaluate_gold_records(
        gold_records=gold_records,
        artifacts=artifacts,
        output_dir=out_dir,
        llm_config=llm_config,
        max_items=args.max_items,
        dry_run=args.dry_run,
        overwrite=not args.no_overwrite,
    )

    if args.dry_run:
        print(f"[eval] dry run complete; contexts -> {summary['contexts_path']}")
    else:
        print(f"[eval] complete; summary -> {out_dir / 'summary.json'}")
        print(f"[eval] report -> {out_dir / 'evaluation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
