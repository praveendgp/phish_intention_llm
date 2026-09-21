#!/usr/bin/env python
"""Pre-flight check: Ollama, models, VLM capability and dataset layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishintentionllm.config import FRAMEWORK_ROLES, MANIFEST_ROLES, load_config
from phishintentionllm.datasets import DatasetRegistry
from phishintentionllm.pipeline import PhishIntentionLLM
from phishintentionllm.rag.knowledge_base import get_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="PhishIntentionLLM setup check")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    print(f"\n=== PhishIntentionLLM setup check ===\nconfig: {cfg.path}\n")

    print("[1/5] Knowledge base")
    print(f"      {get_knowledge_base().stats()}\n")

    print("[2/5] Datasets")
    for info in DatasetRegistry(cfg).available():
        print(f"      [{'OK ' if info['exists'] else 'MISSING'}] "
              f"{info['source']:<10} {info['path']}")
    print()

    print("[3/5] Vision configuration")
    bad = [s.role for s in cfg.all_models() if not s.vision]
    if bad:
        print(f"      [FAIL] these roles are not vision-enabled: {bad}")
        print("             Every agent must analyse the screenshot with a VLM.")
        return 1
    print("      [OK ] every configured role declares vision: true\n")

    print("[4/5] Ollama")
    framework = PhishIntentionLLM(cfg)
    report = framework.preflight()
    if not report["alive"]:
        print(f"      [FAIL] cannot reach {report['host']} - run 'ollama serve'")
        return 1
    print(f"      [OK ] {report['host']}\n")

    print("[5/5] Models")
    for group, roles in (("manifest stage (annotators)", MANIFEST_ROLES),
                         ("framework stage (5 layers)", FRAMEWORK_ROLES)):
        print(f"      -- {group}")
        for role in roles:
            info = report["models"].get(role, {})
            mark = "OK " if info.get("ready") else "PULL"
            vision = info.get("detected_vision")
            note = ("" if vision is not False else "   <-- NOT MULTIMODAL")
            print(f"         [{mark}] {role:<20} {info.get('model', '?')}{note}")

    if report["missing"]:
        print("\n      Pull the missing models:")
        for model in report["missing"]:
            print(f"        ollama pull {model}")
        return 1
    if report["non_vision"]:
        print(f"\n      [FAIL] not multimodal: {report['non_vision']}")
        print("             Replace them in config.yaml with vision models.")
        return 1

    print("\nEverything is ready.")
    print("  1) build the manifest : python scripts/build_manifest.py --n 20")
    print("  2) run the framework  : python scripts/run_predictions.py --scope manifest")
    print("  3) evaluate           : python scripts/run_evaluation.py")
    print("  or launch the UI      : ./run_ui.sh\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
