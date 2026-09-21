#!/usr/bin/env python3
"""Adjudicate local Ollama annotation disagreements with a third vision model.

Primary annotators are provider-neutral (for example Gemma and MiniCPM-V).
Only disputed labels are decided by the adjudicator. Labels on which the two
primary annotators agree are preserved. Every adjudicated row remains marked
for human review.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from phishintention.annotators import OllamaAdjudicator

LABELS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]
TRUE_VALUES = {"1", "true", "yes", "y"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a local Ollama vision model to adjudicate only disputed labels."
    )
    parser.add_argument("--agreement", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotation-a", required=True)
    parser.add_argument("--annotation-b", required=True)
    parser.add_argument("--provider-a", required=True)
    parser.add_argument("--provider-b", required=True)
    parser.add_argument("--model", default="mistral-small3.1:24b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--cache",
        default="data/annotations/local/ollama_mistral_adjudications.jsonl",
    )
    parser.add_argument(
        "--failure-log",
        default="data/annotations/local/ollama_mistral_failures.jsonl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess rows already present in the adjudication cache.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def provider_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    if not cleaned:
        raise SystemExit("Provider name cannot be empty.")
    return cleaned


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in TRUE_VALUES


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"JSONL file not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON in {path}, line {line_number}: {error}") from error
        sample_id = str(item.get("sample_id", "")).strip()
        if sample_id:
            records[sample_id] = item
    return records


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def annotation(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("annotation", record)
    return value if isinstance(value, dict) else {}


def decision(ann: dict[str, Any], label: str) -> dict[str, Any]:
    value = ann.get(label, {})
    if isinstance(value, dict):
        return value
    return {"present": value, "confidence": "", "evidence": []}


def evidence_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return ""


def disagreement_labels(row: pd.Series) -> list[str]:
    labels: list[str] = []
    for label in LABELS:
        column = f"agree_{label}"
        if column not in row.index or not as_bool(row[column]):
            labels.append(label)
    return labels


def ensure_final_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.astype(object)
    defaults = {
        "adjudicator_model": "",
        "adjudication_source": "",
        "requires_human_review": "False",
        "disagreement_labels": "",
        "final_image_quality": "",
        "final_exclusion_reason": "",
        "final_annotation_notes": "",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    for label in LABELS:
        for suffix in ("", "_confidence", "_evidence"):
            column = f"final_{label}{suffix}"
            if column not in frame.columns:
                frame[column] = ""
    return frame


def merge_agreed_labels(
    frame: pd.DataFrame,
    index: int,
    row: pd.Series,
    ann_a: dict[str, Any],
    ann_b: dict[str, Any],
    disputed: set[str],
) -> None:
    for label in LABELS:
        if label in disputed:
            continue
        dec_a = decision(ann_a, label)
        dec_b = decision(ann_b, label)
        present_a = as_bool(dec_a.get("present", False))
        present_b = as_bool(dec_b.get("present", False))
        if present_a != present_b:
            raise ValueError(f"Agreement CSV says {label} agrees, but JSONL values differ.")
        frame.at[index, f"final_{label}"] = str(int(present_a))
        frame.at[index, f"final_{label}_confidence"] = ""
        evidence = evidence_text(dec_a.get("evidence", []))
        if not evidence:
            evidence = evidence_text(dec_b.get("evidence", []))
        frame.at[index, f"final_{label}_evidence"] = evidence


def adjudication_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        value = result.model_dump(mode="json")
    elif isinstance(result, dict):
        value = result
    else:
        raise TypeError(f"Unsupported adjudicator result type: {type(result).__name__}")
    if not isinstance(value, dict):
        raise TypeError("Adjudicator result is not a JSON object.")
    return value


def call_adjudicator(
    adjudicator: OllamaAdjudicator,
    image_path: Path,
    ann_a: dict[str, Any],
    ann_b: dict[str, Any],
    disputed: list[str],
) -> tuple[Any, dict[str, Any]]:
    """Support both provider-neutral and older parameter names."""
    try:
        return adjudicator.annotate(
            image_path=image_path,
            annotation_a=ann_a,
            annotation_b=ann_b,
            disagreement_labels=disputed,
        )
    except TypeError as error:
        message = str(error)
        if "unexpected keyword argument" not in message:
            raise
        return adjudicator.annotate(
            image_path=image_path,
            openai_annotation=ann_a,
            gemini_annotation=ann_b,
            disagreement_labels=disputed,
        )


def write_latest_failures(path: Path, failures: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(failures):
            handle.write(json.dumps(failures[sample_id], ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_arguments()
    agreement_path = resolve_path(args.agreement)
    output_path = resolve_path(args.output)
    annotation_a_path = resolve_path(args.annotation_a)
    annotation_b_path = resolve_path(args.annotation_b)
    cache_path = resolve_path(args.cache)
    failure_path = resolve_path(args.failure_log)
    provider_a = provider_name(args.provider_a)
    provider_b = provider_name(args.provider_b)

    if not agreement_path.exists():
        raise SystemExit(f"Agreement CSV not found: {agreement_path}")

    source_frame = pd.read_csv(agreement_path, dtype=str, keep_default_na=False)
    frame = ensure_final_columns(source_frame.copy())
    records_a = load_jsonl(annotation_a_path)
    records_b = load_jsonl(annotation_b_path)
    cached = load_jsonl(cache_path) if cache_path.exists() else {}
    existing_failures = load_jsonl(failure_path) if failure_path.exists() else {}

    if "sample_id" not in frame.columns or "image_path" not in frame.columns:
        raise SystemExit("Agreement CSV must contain sample_id and image_path columns.")

    selected_indices: list[int] = []
    for index, row in frame.iterrows():
        disputed = disagreement_labels(row)
        quality_disagrees = "agree_image_quality" in row.index and not as_bool(row["agree_image_quality"])
        required = disputed or quality_disagrees
        sample_id = str(row["sample_id"])
        if required and (args.force or sample_id not in cached):
            selected_indices.append(index)

    if args.limit is not None:
        selected_indices = selected_indices[: args.limit]

    print(f"Rows selected for local adjudication: {len(selected_indices)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.resolve() != agreement_path.resolve():
        backup = output_path.with_suffix(output_path.suffix + ".backup")
        shutil.copy2(output_path, backup)
        print(f"Existing output backup: {backup}")

    adjudicator = OllamaAdjudicator(model=args.model)
    successes = 0
    failures = dict(existing_failures)

    for position, index in enumerate(selected_indices, 1):
        row = frame.loc[index]
        sample_id = str(row["sample_id"])
        image_path = Path(str(row["image_path"])).expanduser()
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_path = image_path.resolve()
        if sample_id not in records_a or sample_id not in records_b:
            error = f"Missing primary annotation for sample {sample_id}"
            failures[sample_id] = {"sample_id": sample_id, "error_type": "KeyError", "error": error}
            print(f"[{position}/{len(selected_indices)}] {error}", file=sys.stderr)
            continue

        ann_a = annotation(records_a[sample_id])
        ann_b = annotation(records_b[sample_id])
        disputed = disagreement_labels(row)
        disputed_set = set(disputed)
        quality_disagrees = "agree_image_quality" in row.index and not as_bool(row["agree_image_quality"])

        print(
            f"\n[{position}/{len(selected_indices)}] LOCAL ADJUDICATION"
            f"\nSample ID: {sample_id}"
            f"\nImage: {image_path}"
            f"\nProviders: {provider_a}, {provider_b}"
            f"\nDisputed labels: {', '.join(disputed) or 'none'}"
            f"\nImage-quality disagreement: {quality_disagrees}"
            f"\nModel: {args.model}",
            flush=True,
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            merge_agreed_labels(frame, index, row, ann_a, ann_b, disputed_set)
            result, metadata = call_adjudicator(
                adjudicator=adjudicator,
                image_path=image_path,
                ann_a=ann_a,
                ann_b=ann_b,
                disputed=disputed,
            )
            payload = adjudication_payload(result)

            for label in disputed:
                dec = decision(payload, label)
                frame.at[index, f"final_{label}"] = str(int(as_bool(dec.get("present", False))))
                confidence = dec.get("confidence", "")
                frame.at[index, f"final_{label}_confidence"] = "" if confidence is None else str(confidence)
                frame.at[index, f"final_{label}_evidence"] = evidence_text(dec.get("evidence", []))

            quality_a = str(ann_a.get("image_quality", "")).strip()
            quality_b = str(ann_b.get("image_quality", "")).strip()
            if quality_disagrees:
                frame.at[index, "final_image_quality"] = str(payload.get("image_quality", ""))
            else:
                frame.at[index, "final_image_quality"] = quality_a or quality_b

            frame.at[index, "final_exclusion_reason"] = str(payload.get("exclusion_reason", ""))
            frame.at[index, "final_annotation_notes"] = str(
                payload.get("adjudication_summary", payload.get("annotation_notes", ""))
            )
            frame.at[index, "disagreement_labels"] = " | ".join(disputed)
            frame.at[index, "adjudicator_model"] = str(metadata.get("model", args.model))
            frame.at[index, "adjudication_source"] = "ollama_mistral_tiebreaker"
            frame.at[index, "requires_human_review"] = "True"

            cache_record = {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "source": str(row.get("source", "")),
                "provider_a": provider_a,
                "provider_b": provider_b,
                "disagreement_labels": disputed,
                "image_quality_disagreement": quality_disagrees,
                "annotation": payload,
                "metadata": metadata,
            }
            append_jsonl(cache_path, cache_record)
            cached[sample_id] = cache_record
            failures.pop(sample_id, None)
            successes += 1
            frame.to_csv(output_path, index=False)
            print("Adjudication saved.", flush=True)

        except Exception as error:
            failures[sample_id] = {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "provider_a": provider_a,
                "provider_b": provider_b,
                "model": args.model,
                "disagreement_labels": disputed,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            frame.at[index, "adjudication_source"] = "ollama_mistral_failed"
            frame.at[index, "requires_human_review"] = "True"
            frame.at[index, "adjudicator_model"] = args.model
            frame.at[index, "final_annotation_notes"] = f"Local adjudication failed: {error}"
            frame.to_csv(output_path, index=False)
            print(f"Adjudication failed: {error}", file=sys.stderr, flush=True)

        write_latest_failures(failure_path, failures)
        if args.sleep > 0 and position < len(selected_indices):
            time.sleep(args.sleep)

    # Preserve direct agreement rows and write output even when no dispute was selected.
    for index, row in frame.iterrows():
        if disagreement_labels(row):
            continue
        if "agree_image_quality" in row.index and not as_bool(row["agree_image_quality"]):
            continue
        sample_id = str(row["sample_id"])
        if sample_id not in records_a or sample_id not in records_b:
            continue
        ann_a = annotation(records_a[sample_id])
        ann_b = annotation(records_b[sample_id])
        merge_agreed_labels(frame, index, row, ann_a, ann_b, set())
        quality_a = str(ann_a.get("image_quality", "")).strip()
        quality_b = str(ann_b.get("image_quality", "")).strip()
        frame.at[index, "final_image_quality"] = quality_a or quality_b
        if not str(frame.at[index, "adjudication_source"]).strip():
            frame.at[index, "adjudication_source"] = "local_primary_agreement"
        frame.at[index, "requires_human_review"] = "False"
        if not str(frame.at[index, "final_annotation_notes"]).strip():
            frame.at[index, "final_annotation_notes"] = "Primary local annotators agree."

    frame.to_csv(output_path, index=False)
    write_latest_failures(failure_path, failures)

    print("\nLocal adjudication complete.")
    print(f"Selected: {len(selected_indices)}")
    print(f"Succeeded: {successes}")
    print(f"Failed: {len(failures)}")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_path}")
    print(f"Failure log: {failure_path}")
    print("Every Mistral-resolved row still requires human review.")


if __name__ == "__main__":
    main()
