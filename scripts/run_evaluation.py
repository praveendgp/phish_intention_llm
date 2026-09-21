#!/usr/bin/env python
"""Score framework predictions against the annotator-generated manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import AnnotationStore
from phishintentionllm.config import load_config
from phishintentionllm.evaluation import Evaluator


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PhishIntentionLLM")
    parser.add_argument("--run", default=None, help="path to a runs/run_*.json file")
    parser.add_argument("--reference", choices=["manifest", "ground_truth"],
                        default=None, help="which label set to score against")
    parser.add_argument("--verified-only", action="store_true",
                        help="use only human-verified manifest rows")
    parser.add_argument("--compare", nargs=2, metavar=("FRAMEWORK", "BASELINE"))
    parser.add_argument("--errors", action="store_true",
                        help="print the per-sample error analysis")
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = Evaluator(cfg, AnnotationStore(cfg))

    if args.compare:
        payload = evaluator.compare_runs(*args.compare, args.reference)
        print(json.dumps(payload, indent=2, default=str)[:4000])
        return 0

    payload = evaluator.evaluate_run(args.run, args.reference, args.verified_only)
    if "error" in payload:
        print(f"ERROR: {payload['error']}")
        print(json.dumps(payload, indent=2))
        return 1

    report = payload["report"]
    print(f"\n=== Evaluation vs the {payload['reference']} "
          f"({payload['n_evaluated']} samples) ===")
    print(f"  Precision (micro) : {report['precision_micro']:.4f}")
    print(f"  Recall    (micro) : {report['recall_micro']:.4f}")
    print(f"  F1        (micro) : {report['f1_micro']:.4f}")
    print(f"  Accuracy  (micro) : {report['accuracy_micro']:.4f}")
    print(f"  Overall accuracy  : {report['overall_accuracy']:.4f}")

    print("\n  Accuracy by complexity:")
    for key, value in report["acc_by_complexity"].items():
        print(f"    {key:<18} {value:.4f}")

    print("\n  Set-level agreement with the reference:")
    for key, value in payload["agreement"].items():
        print(f"    {key:<16} {value}")

    print("\n  Per class:")
    for category, values in report["per_class"].items():
        print(f"    {category:<34} P={values['precision']:.4f} "
              f"R={values['recall']:.4f} F1={values['f1']:.4f} "
              f"Acc={values['accuracy']:.4f}")

    quality = evaluator.manifest_quality()
    if quality.get("total"):
        print(f"\n  Reference quality: {quality['total']} rows, "
              f"IAA={quality['inter_annotator_agreement']:.1%}, "
              f"tie-breaks={quality['tie_breaks']}, "
              f"needs review={quality['needs_review']}, "
              f"human verified={quality['human_verified']}")

    if args.errors:
        print("\n  Per-sample analysis:")
        for row in evaluator.per_sample(args.run, args.reference):
            if row["outcome"] != "exact":
                print(f"    [{row['outcome']:<7}] {row['sample_id'][-40:]:<40} "
                      f"missed: {row['missed']:<24} extra: {row['extra']}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str),
                                  encoding="utf-8")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
