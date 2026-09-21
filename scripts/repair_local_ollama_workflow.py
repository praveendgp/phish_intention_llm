#!/usr/bin/env python3
"""Repair the local-only Ollama annotation workflow after cloud cleanup.

Repairs:
1. Rebuilds annotators/__init__.py without dangling parentheses.
2. Replaces calculate_annotation_agreement.py with a provider-neutral implementation.
3. Renames OpenAI/Gemini parameter names and prompt wording in ollama_adjudicator.py.
4. Renames provider-specific variables in consensus.py without changing behaviour.
5. Removes stale egg-info generated metadata.
6. Backs up every changed file before applying changes.
7. Runs compileall and pytest unless --skip-tests is supplied.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AGREEMENT_SCRIPT = r'''from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

TRUE_VALUES = {"1", "true", "yes", "y"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two independent local Ollama annotation JSONL files."
    )
    parser.add_argument("--annotation-a", required=True)
    parser.add_argument("--annotation-b", required=True)
    parser.add_argument("--provider-a", required=True)
    parser.add_argument("--provider-b", required=True)
    parser.add_argument("--output-dir", default="data/annotations/local")
    parser.add_argument("--create-image-links", action="store_true")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Annotation file not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON: {path}, line {line_number}: {error}") from error
        sample_id = str(item.get("sample_id", "")).strip()
        if not sample_id:
            raise SystemExit(f"Missing sample_id: {path}, line {line_number}")
        records[sample_id] = item
    return records


def annotation_of(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation", record)
    return value if isinstance(value, dict) else {}


def decision_of(annotation: dict[str, Any], label: str) -> dict[str, Any]:
    value = annotation.get(label, {})
    if isinstance(value, dict):
        return value
    return {"present": value}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in TRUE_VALUES


def evidence_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def model_of(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("model", ""))
    return ""


def safe_provider(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    if not cleaned:
        raise SystemExit("Provider name cannot be empty.")
    return cleaned


def create_image_links(queue: pd.DataFrame, output_dir: Path) -> int:
    image_dir = output_dir / "adjudication_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for position, row in queue.reset_index(drop=True).iterrows():
        source = Path(str(row.get("image_path", ""))).expanduser()
        if not source.exists():
            print(f"Warning: image not found: {source}")
            continue
        target = image_dir / f"{position + 1:05d}_{row['sample_id']}_{source.name}"
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
        created += 1
    return created


def main() -> None:
    args = parse_arguments()
    annotation_a_path = resolve_path(args.annotation_a)
    annotation_b_path = resolve_path(args.annotation_b)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_a = safe_provider(args.provider_a)
    provider_b = safe_provider(args.provider_b)
    if provider_a == provider_b:
        raise SystemExit("Provider A and provider B must be different names.")

    records_a = load_records(annotation_a_path)
    records_b = load_records(annotation_b_path)
    shared_ids = sorted(set(records_a) & set(records_b))
    only_a = sorted(set(records_a) - set(records_b))
    only_b = sorted(set(records_b) - set(records_a))
    if not shared_ids:
        raise SystemExit("The annotation files have no shared sample IDs.")

    rows: list[dict[str, Any]] = []
    per_label = defaultdict(lambda: {"agree": 0, "total": 0})
    exact_agreement = 0

    for sample_id in shared_ids:
        record_a = records_a[sample_id]
        record_b = records_b[sample_id]
        ann_a = annotation_of(record_a)
        ann_b = annotation_of(record_b)
        image_path = str(record_a.get("image_path") or record_b.get("image_path") or "")
        source = str(record_a.get("source") or record_b.get("source") or "")
        quality_a = str(ann_a.get("image_quality", "")).strip().lower()
        quality_b = str(ann_b.get("image_quality", "")).strip().lower()

        row: dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_path,
            "source": source,
            "provider_a": provider_a,
            "provider_b": provider_b,
            "provider_a_model": model_of(record_a),
            "provider_b_model": model_of(record_b),
            f"{provider_a}_image_quality": quality_a,
            f"{provider_b}_image_quality": quality_b,
            "agree_image_quality": quality_a == quality_b,
            f"{provider_a}_notes": str(ann_a.get("annotation_notes", "")),
            f"{provider_b}_notes": str(ann_b.get("annotation_notes", "")),
        }

        disagreement_labels: list[str] = []
        all_labels_agree = True
        for label in LABELS:
            decision_a = decision_of(ann_a, label)
            decision_b = decision_of(ann_b, label)
            present_a = as_bool(decision_a.get("present", False))
            present_b = as_bool(decision_b.get("present", False))
            agrees = present_a == present_b
            row[f"{provider_a}_{label}"] = int(present_a)
            row[f"{provider_b}_{label}"] = int(present_b)
            row[f"agree_{label}"] = agrees
            row[f"{provider_a}_{label}_confidence"] = decision_a.get("confidence", "")
            row[f"{provider_b}_{label}_confidence"] = decision_b.get("confidence", "")
            row[f"{provider_a}_{label}_evidence"] = evidence_text(decision_a.get("evidence", []))
            row[f"{provider_b}_{label}_evidence"] = evidence_text(decision_b.get("evidence", []))
            per_label[label]["total"] += 1
            if agrees:
                per_label[label]["agree"] += 1
                row[f"final_{label}"] = int(present_a)
                row[f"final_{label}_confidence"] = ""
                row[f"final_{label}_evidence"] = row[f"{provider_a}_{label}_evidence"]
            else:
                all_labels_agree = False
                disagreement_labels.append(label)
                row[f"final_{label}"] = ""
                row[f"final_{label}_confidence"] = ""
                row[f"final_{label}_evidence"] = ""

        quality_agrees = quality_a == quality_b
        adjudication_required = bool(disagreement_labels) or not quality_agrees
        if all_labels_agree and quality_agrees:
            exact_agreement += 1

        row["disagreement_labels"] = " | ".join(disagreement_labels)
        row["adjudication_required"] = adjudication_required
        row["adjudication_source"] = (
            "local_primary_agreement" if not adjudication_required else "local_disagreement_pending"
        )
        row["requires_human_review"] = adjudication_required
        row["adjudicator_model"] = ""
        row["final_image_quality"] = quality_a if quality_agrees else ""
        row["final_exclusion_reason"] = ""
        row["final_annotation_notes"] = (
            "Primary local annotators agree." if not adjudication_required
            else "Local primary annotators disagree; provisional adjudication required."
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    agreement_path = output_dir / "agreement.csv"
    frame.to_csv(agreement_path, index=False)
    queue = frame[frame["adjudication_required"] == True].copy()  # noqa: E712
    queue_path = output_dir / "adjudication_queue.csv"
    queue.to_csv(queue_path, index=False)

    metrics = {
        "provider_a": provider_a,
        "provider_b": provider_b,
        "provider_a_model": sorted({model_of(records_a[sid]) for sid in shared_ids}),
        "provider_b_model": sorted({model_of(records_b[sid]) for sid in shared_ids}),
        "annotation_a_count": len(records_a),
        "annotation_b_count": len(records_b),
        "shared_count": len(shared_ids),
        "only_annotation_a_count": len(only_a),
        "only_annotation_b_count": len(only_b),
        "exact_agreement_count": exact_agreement,
        "exact_agreement_rate": exact_agreement / len(shared_ids),
        "adjudication_required_count": len(queue),
        "per_label": {
            label: {
                "agreement_count": values["agree"],
                "total": values["total"],
                "agreement_rate": values["agree"] / values["total"] if values["total"] else None,
            }
            for label, values in per_label.items()
        },
        "only_annotation_a_sample_ids": only_a,
        "only_annotation_b_sample_ids": only_b,
    }
    metrics_path = output_dir / "agreement_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    links = 0
    if args.create_image_links:
        links = create_image_links(queue, output_dir)

    print(f"Created: {agreement_path}")
    print(f"Created: {metrics_path}")
    print(f"Created: {queue_path}")
    print(f"Provider A records: {len(records_a)}")
    print(f"Provider B records: {len(records_b)}")
    print(f"Shared records: {len(shared_ids)}")
    print(f"Exact agreements: {exact_agreement}")
    print(f"Adjudication required: {len(queue)}")
    if args.create_image_links:
        print(f"Image links created: {links}")


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair the provider-neutral local Ollama annotation workflow."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def class_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def build_init(root: Path) -> str:
    modules = [
        ("schemas", root / "src/phishintention/annotators/schemas.py"),
        ("ollama_annotator", root / "src/phishintention/annotators/ollama_annotator.py"),
        ("ollama_adjudicator", root / "src/phishintention/annotators/ollama_adjudicator.py"),
    ]
    preferred = {
        "schemas": ["AnnotationOutput", "LabelDecision"],
        "ollama_annotator": ["OllamaAnnotator", "LocalAnnotationOutput"],
        "ollama_adjudicator": ["OllamaAdjudicator", "OllamaAdjudicationOutput"],
    }
    imports: list[str] = ['"""Local Ollama annotation components."""', ""]
    exports: list[str] = []
    for module, path in modules:
        available = class_names(path)
        selected = [name for name in preferred[module] if name in available]
        if not selected:
            continue
        imports.append(f"from .{module} import (")
        imports.extend(f"    {name}," for name in selected)
        imports.append(")")
        imports.append("")
        exports.extend(selected)
    imports.append("__all__ = [")
    imports.extend(f'    "{name}",' for name in exports)
    imports.append("]")
    imports.append("")
    return "\n".join(imports)


def neutralise_adjudicator(text: str) -> str:
    replacements = [
        ("annotations produced by OpenAI and Gemini", "annotations produced by two independent local Ollama models"),
        ("OpenAI annotation:", "Provider A annotation:"),
        ("Gemini annotation:", "Provider B annotation:"),
        ("{openai_annotation}", "{annotation_a}"),
        ("{gemini_annotation}", "{annotation_b}"),
        ("openai_annotation", "annotation_a"),
        ("gemini_annotation", "annotation_b"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def neutralise_consensus(text: str) -> str:
    replacements = [
        ("openai_path", "annotation_a_path"),
        ("gemini_path", "annotation_b_path"),
        ("openai_annotations", "annotations_a"),
        ("gemini_annotations", "annotations_b"),
        ("openai_record", "record_a"),
        ("gemini_record", "record_b"),
        ("openai_image_path", "image_path_a"),
        ("gemini_image_path", "image_path_b"),
        ("openai_quality", "quality_a"),
        ("gemini_quality", "quality_b"),
        ("OpenAI and Gemini", "two independent local Ollama providers"),
        ("OpenAI path", "provider A path"),
        ("Gemini", "provider B"),
        ("OpenAI", "provider A"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def run(command: list[str], cwd: Path) -> int:
    print("\n$", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, text=True, check=False)
    return completed.returncode


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not (root / "src/phishintention/annotators").is_dir():
        raise SystemExit(f"Not a PhishIntentionLLM project root: {root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "migration_backups" / f"local_workflow_repair_{timestamp}"
    init_path = root / "src/phishintention/annotators/__init__.py"
    agreement_path = root / "scripts/calculate_annotation_agreement.py"
    adjudicator_path = root / "src/phishintention/annotators/ollama_adjudicator.py"
    consensus_path = root / "src/phishintention/annotators/consensus.py"
    targets = [path for path in (init_path, agreement_path, adjudicator_path, consensus_path) if path.exists()]

    print("Planned repairs:")
    print("- rebuild annotators/__init__.py")
    print("- replace calculate_annotation_agreement.py with provider-neutral CLI")
    print("- neutralise provider names in ollama_adjudicator.py")
    print("- neutralise provider variable names in consensus.py")
    print("- remove generated *.egg-info directories")

    if args.dry_run:
        print("\nDry run only. No files changed.")
        return

    for path in targets:
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    init_path.write_text(build_init(root), encoding="utf-8")
    agreement_path.write_text(AGREEMENT_SCRIPT, encoding="utf-8")
    if adjudicator_path.exists():
        adjudicator_path.write_text(
            neutralise_adjudicator(adjudicator_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    if consensus_path.exists():
        consensus_path.write_text(
            neutralise_consensus(consensus_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

    removed_egg_info = []
    for path in root.glob("**/*.egg-info"):
        if ".venv" in path.parts:
            continue
        removed_egg_info.append(str(path.relative_to(root)))
        shutil.rmtree(path)

    report = {
        "backup": str(backup),
        "changed": [str(path.relative_to(root)) for path in targets],
        "removed_egg_info": removed_egg_info,
    }
    (root / "local_workflow_repair_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    failures = 0
    if not args.skip_tests:
        commands = [
            [sys.executable, "-m", "py_compile", str(init_path), str(agreement_path)],
            [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
            [sys.executable, "-m", "pytest", "-q"],
        ]
        for command in commands:
            failures += int(run(command, root) != 0)

    print("\nRepair complete.")
    print("Backup:", backup)
    print("Report:", root / "local_workflow_repair_report.json")
    print("\nVerify CLI:")
    print("python scripts/calculate_annotation_agreement.py --help")
    print("\nThen run:")
    print("python scripts/calculate_annotation_agreement.py \\")
    print("  --annotation-a data/annotations/ollama_gemma_annotations.jsonl \\")
    print("  --annotation-b data/annotations/ollama_minicpm_annotations.jsonl \\")
    print("  --provider-a gemma --provider-b minicpm \\")
    print("  --output-dir data/annotations/local --create-image-links")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
