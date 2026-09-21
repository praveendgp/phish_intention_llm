#!/usr/bin/env python
"""Build the manifest with the annotator ensemble.

    Annotator A + Annotator B  ->  tie-breaker (on dispute)  ->  finaliser
                              ->  data/outputs/manifest.jsonl

The manifest is the reference label set used to evaluate the framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import AnnotationStore, ManifestBuilder
from phishintentionllm.config import load_config
from phishintentionllm.datasets import DatasetRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the annotation manifest")
    parser.add_argument("--n", type=int, default=20, help="number of samples")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="putra phishiris (default: all available)")
    parser.add_argument("--strategy", choices=["random", "first"], default="random")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-annotate samples already in the manifest")
    parser.add_argument("--csv", default=None, help="also export a CSV copy")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    registry = DatasetRegistry(cfg)
    store = AnnotationStore(cfg)

    sources = args.sources or [d["source"] for d in registry.available()
                               if d["exists"]]
    if not sources:
        print("No dataset found under data/raw/. Nothing to annotate.")
        return 1

    if args.strategy == "random":
        pool = registry.sample(args.n * 3, sources=sources, stratify=True)
    else:
        pool = registry.load_all(limit_per_source=args.n, sources=sources)

    if not args.overwrite:
        done = set(store.manifest_ids())
        pool = [s for s in pool if s.sample_id not in done]

    selection = pool[: args.n]
    if not selection:
        print("Nothing left to annotate - the manifest already covers these samples.")
        return 0

    builder = ManifestBuilder(cfg, store=store)
    print(f"Annotating {len(selection)} sample(s) from {sources}")
    print(f"  Annotator A : {builder.annotator_a.spec.model}")
    print(f"  Annotator B : {builder.annotator_b.spec.model}")
    print(f"  Tie-breaker : {builder.tiebreaker.spec.model}")
    print(f"  Finaliser   : {builder.finalizer.spec.model}\n")

    def progress(index, total, record):
        labels = ", ".join(record.labels) or "-"
        flag = "ERR" if record.error else ("TB " if record.tie_break_used else "OK ")
        print(f"  [{index:>4}/{total}] {flag} {record.sample_id[-40:]:<40} "
              f"-> {labels:<34} ({record.agreement}, {record.elapsed:.1f}s)")

    payload = builder.build(selection, on_progress=progress,
                            overwrite=args.overwrite)

    print("\n=== Manifest run summary ===")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"},
                     indent=2, default=str))
    print(f"\nManifest file: {store.manifest_path}")
    print(f"Total records: {len(store.manifest_ids())}")

    if args.csv:
        print(f"CSV export   : {store.export_manifest_csv(args.csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
