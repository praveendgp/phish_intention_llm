#!/usr/bin/env python
"""Export the reviewed manual annotations as the human ground-truth dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.annotation import AnnotationStore, ManualAnnotationManager
from phishintentionllm.config import load_config
from phishintentionllm.datasets import DatasetRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ground-truth dataset")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    store = AnnotationStore(cfg)
    manager = ManualAnnotationManager(cfg, store)

    rows = manager.build_ground_truth(DatasetRegistry(cfg).load_all(),
                                      require_review=not args.allow_unreviewed)
    print(f"Wrote {store.write_ground_truth(rows)} record(s) to {store.gt_path}")
    print(json.dumps(manager.distribution("ground_truth"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
