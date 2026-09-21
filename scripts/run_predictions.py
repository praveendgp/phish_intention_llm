#!/usr/bin/env python
"""Run the five-layer framework (or the single-agent baseline) over samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import AnnotationStore
from phishintentionllm.config import load_config
from phishintentionllm.datasets import DatasetRegistry
from phishintentionllm.pipeline import BatchRunner, PhishIntentionLLM


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate framework predictions")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--mode", choices=["framework", "single"], default="framework")
    parser.add_argument("--scope", choices=["manifest", "random", "first"],
                        default="manifest",
                        help="'manifest' restricts to samples that have reference labels")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    registry = DatasetRegistry(cfg)
    store = AnnotationStore(cfg)

    sources = args.sources or [d["source"] for d in registry.available()
                               if d["exists"]]
    if not sources:
        print("No dataset found under data/raw/.")
        return 1

    manifest_ids = set(store.manifest_ids())
    if args.scope == "manifest":
        if not manifest_ids:
            print("The manifest is empty. Run scripts/build_manifest.py first, "
                  "or use --scope random.")
            return 1
        pool = [s for s in registry.load_all(sources=sources)
                if s.sample_id in manifest_ids]
    elif args.scope == "random":
        pool = registry.sample(args.n * 3, sources=sources, stratify=True)
    else:
        pool = registry.load_all(limit_per_source=args.n, sources=sources)

    if args.skip_done:
        predicted = set(store.prediction_map())
        pool = [s for s in pool if s.sample_id not in predicted]

    selection = pool[: args.n]
    if not selection:
        print("Nothing left to predict.")
        return 0

    covered = sum(1 for s in selection if s.sample_id in manifest_ids)
    print(f"Predicting {len(selection)} sample(s) with the {args.mode} pipeline "
          f"({covered} have manifest reference labels)\n")

    runner = BatchRunner(cfg, PhishIntentionLLM(cfg), store)

    def progress(index, total, result):
        labels = ", ".join(i.short for i in result.intentions) or "-"
        flag = "ERR" if result.error else "OK "
        print(f"  [{index:>4}/{total}] {flag} {result.sample_id[-40:]:<40} "
              f"-> {labels:<14} ({result.vision_calls} VLM calls, "
              f"{result.elapsed:.1f}s)")

    payload = runner.run(selection, mode=args.mode, on_progress=progress)

    print("\n=== Prediction run summary ===")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
