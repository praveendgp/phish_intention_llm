from __future__ import annotations

import argparse
import json
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

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from phishintention.annotators import GroqAdjudicator
from phishintention.annotators.consensus import LABELS, load_jsonl


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Groq as a third-model tie-breaker for OpenAI and "
            "Gemini annotation disagreements."
        )
    )
    parser.add_argument(
        "--agreement",
        default="data/annotations/agreement.csv",
    )
    parser.add_argument(
        "--openai",
        default="data/annotations/openai_annotations.jsonl",
    )
    parser.add_argument(
        "--gemini",
        default="data/annotations/gemini_annotations.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/final_agreement.csv",
    )
    parser.add_argument(
        "--groq-jsonl",
        default="data/annotations/groq_adjudications.jsonl",
    )
    parser.add_argument(
        "--failure-jsonl",
        default="data/annotations/groq_adjudication_failures.jsonl",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=20.0,
        help="Delay after each new Groq request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard the existing Groq cache and run again.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry sample IDs already present in the failure file.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_binary(value: Any) -> int:
    return int(as_bool(value))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            print(
                f"Warning: skipped invalid JSON in {path}, "
                f"line {line_number}: {error}",
                file=sys.stderr,
            )
            continue
        sample_id = item.get("sample_id")
        if sample_id:
            records[str(sample_id)] = item
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def disagreement_labels_for(row: Any) -> list[str]:
    return [
        label
        for label in LABELS
        if not as_bool(getattr(row, f"agree_{label}", False))
    ]


def agreed_label(row: Any, label: str) -> int:
    openai_value = as_binary(getattr(row, f"openai_{label}"))
    gemini_value = as_binary(getattr(row, f"gemini_{label}"))
    if openai_value != gemini_value:
        raise ValueError(f"OpenAI and Gemini disagree on {label}")
    return openai_value


def filter_annotation(
    annotation: dict[str, Any],
    disagreement_labels: list[str],
    include_all_labels: bool,
) -> dict[str, Any]:
    selected = list(LABELS) if include_all_labels else disagreement_labels
    filtered = {
        label: annotation[label]
        for label in selected
        if label in annotation
    }
    filtered["image_quality"] = annotation.get("image_quality", "")
    filtered["exclusion_reason"] = annotation.get("exclusion_reason", "")
    filtered["annotation_notes"] = annotation.get("annotation_notes", "")
    return filtered


def base_final_row(
    row: Any,
    sample_id: str,
    image_path: str,
    disagreements: list[str],
) -> dict[str, Any]:
    final_row: dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": image_path,
        "source": str(getattr(row, "source", "")),
        "openai_model": str(getattr(row, "openai_model", "")),
        "gemini_model": str(getattr(row, "gemini_model", "")),
        "groq_model": "",
        "adjudication_source": "",
        "requires_human_review": False,
        "disagreement_labels": " | ".join(disagreements),
        "final_image_quality": "",
        "final_exclusion_reason": "",
        "final_annotation_notes": "",
    }
    for label in LABELS:
        final_row[f"final_{label}"] = ""
        final_row[f"final_{label}_confidence"] = ""
        final_row[f"final_{label}_evidence"] = ""
    return final_row


def populate_direct_agreement(final_row: dict[str, Any], row: Any) -> None:
    for label in LABELS:
        final_row[f"final_{label}"] = agreed_label(row, label)
        openai_confidence = getattr(row, f"openai_{label}_confidence", "")
        gemini_confidence = getattr(row, f"gemini_{label}_confidence", "")
        final_row[f"final_{label}_confidence"] = (
            f"openai={openai_confidence}; gemini={gemini_confidence}"
        )

        evidence_parts: list[str] = []
        openai_evidence = str(
            getattr(row, f"openai_{label}_evidence", "")
        ).strip()
        gemini_evidence = str(
            getattr(row, f"gemini_{label}_evidence", "")
        ).strip()
        if openai_evidence:
            evidence_parts.append(f"OpenAI: {openai_evidence}")
        if gemini_evidence:
            evidence_parts.append(f"Gemini: {gemini_evidence}")
        final_row[f"final_{label}_evidence"] = " || ".join(evidence_parts)


def populate_groq_result(
    final_row: dict[str, Any],
    annotation: dict[str, Any],
) -> None:
    for label in LABELS:
        decision = annotation.get(label, {})
        if not isinstance(decision, dict):
            decision = {}
        final_row[f"final_{label}"] = int(
            bool(decision.get("present", False))
        )
        final_row[f"final_{label}_confidence"] = decision.get(
            "confidence", ""
        )
        evidence = decision.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []
        final_row[f"final_{label}_evidence"] = " | ".join(
            str(item).strip() for item in evidence if str(item).strip()
        )


def save_final_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def record_failure(
    failure_path: Path,
    sample_id: str,
    image_path: str,
    source: str,
    model: str,
    disagreements: list[str],
    error: Exception,
) -> None:
    append_jsonl(
        failure_path,
        {
            "sample_id": sample_id,
            "image_path": image_path,
            "source": source,
            "model": model,
            "disagreement_labels": disagreements,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def main() -> None:
    arguments = parse_arguments()

    agreement_path = resolve_path(arguments.agreement)
    openai_path = resolve_path(arguments.openai)
    gemini_path = resolve_path(arguments.gemini)
    output_path = resolve_path(arguments.output)
    groq_cache_path = resolve_path(arguments.groq_jsonl)
    failure_path = resolve_path(arguments.failure_jsonl)

    if not agreement_path.exists():
        raise SystemExit(
            f"agreement.csv not found: {agreement_path}\n"
            "Run calculate_annotation_agreement.py first."
        )
    if not openai_path.exists():
        raise SystemExit(f"OpenAI annotations not found: {openai_path}")
    if not gemini_path.exists():
        raise SystemExit(f"Gemini annotations not found: {gemini_path}")

    agreement = pd.read_csv(agreement_path).fillna("")
    if arguments.limit is not None:
        agreement = agreement.head(arguments.limit)
    if agreement.empty:
        raise SystemExit("The selected agreement dataset is empty.")

    required = {
        "sample_id",
        "image_path",
        "openai_quality",
        "gemini_quality",
    }
    for label in LABELS:
        required.update(
            {
                f"openai_{label}",
                f"gemini_{label}",
                f"agree_{label}",
            }
        )
    missing = sorted(required - set(agreement.columns))
    if missing:
        raise SystemExit(
            "agreement.csv is missing required columns:\n- "
            + "\n- ".join(missing)
        )

    openai_records = load_jsonl(openai_path)
    gemini_records = load_jsonl(gemini_path)

    if arguments.force:
        groq_cache_path.parent.mkdir(parents=True, exist_ok=True)
        groq_cache_path.write_text("", encoding="utf-8")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text("", encoding="utf-8")
        groq_cache: dict[str, dict[str, Any]] = {}
        failed_ids: set[str] = set()
    else:
        groq_cache = read_jsonl(groq_cache_path)
        failed_ids = (
            set()
            if arguments.retry_failed
            else set(read_jsonl(failure_path).keys())
        )

    adjudicator = GroqAdjudicator(model=arguments.model)
    final_rows: list[dict[str, Any]] = []

    counters = {
        "direct": 0,
        "cached": 0,
        "new": 0,
        "failed": 0,
        "previous_failed": 0,
    }

    total = len(agreement)

    for position, row in enumerate(
        agreement.itertuples(index=False),
        start=1,
    ):
        sample_id = str(row.sample_id)
        image_object = Path(str(row.image_path)).expanduser()
        if not image_object.is_absolute():
            image_object = PROJECT_ROOT / image_object
        image_object = image_object.resolve()
        image_path = str(image_object)

        disagreements = disagreement_labels_for(row)
        openai_quality = str(row.openai_quality).strip().lower()
        gemini_quality = str(row.gemini_quality).strip().lower()
        quality_disagreement = openai_quality != gemini_quality
        both_usable = openai_quality == "usable" and gemini_quality == "usable"
        needs_groq = bool(disagreements) or quality_disagreement or not both_usable

        final_row = base_final_row(
            row,
            sample_id,
            image_path,
            disagreements,
        )

        if not needs_groq:
            populate_direct_agreement(final_row, row)
            final_row["adjudication_source"] = "openai_gemini_agreement"
            final_row["requires_human_review"] = False
            final_row["final_image_quality"] = "usable"
            final_row["final_annotation_notes"] = (
                "OpenAI and Gemini agreed on all intention labels."
            )
            counters["direct"] += 1
            final_rows.append(final_row)
            save_final_rows(final_rows, output_path)
            print(
                f"\n[{position}/{total}]\n"
                f"Image: {image_path}\n"
                "Source: openai_gemini_agreement\n"
                "Final labels: "
                + str(
                    {
                        label: final_row[f"final_{label}"]
                        for label in LABELS
                    }
                ),
                flush=True,
            )
            continue

        if sample_id in groq_cache:
            cache_record = groq_cache[sample_id]
            annotation = cache_record.get("annotation", {})
            metadata = cache_record.get("metadata", {})
            populate_groq_result(final_row, annotation)
            final_row["groq_model"] = metadata.get(
                "model", adjudicator.model
            )
            final_row["adjudication_source"] = "groq_tiebreaker_cached"
            final_row["requires_human_review"] = True
            final_row["final_image_quality"] = annotation.get(
                "image_quality", ""
            )
            final_row["final_exclusion_reason"] = annotation.get(
                "exclusion_reason", ""
            )
            final_row["final_annotation_notes"] = annotation.get(
                "adjudication_summary",
                annotation.get("annotation_notes", ""),
            )
            counters["cached"] += 1
            final_rows.append(final_row)
            save_final_rows(final_rows, output_path)
            print(
                f"\n[{position}/{total}]\n"
                f"Image: {image_path}\n"
                "Source: groq_tiebreaker_cached\n"
                "Final labels: "
                + str(
                    {
                        label: final_row[f"final_{label}"]
                        for label in LABELS
                    }
                ),
                flush=True,
            )
            continue

        if sample_id in failed_ids:
            final_row["groq_model"] = adjudicator.model
            final_row["adjudication_source"] = "groq_failed_previous_run"
            final_row["requires_human_review"] = True
            final_row["final_annotation_notes"] = (
                "Groq failed during a previous run. Use --retry-failed "
                "or complete human review."
            )
            counters["previous_failed"] += 1
            final_rows.append(final_row)
            save_final_rows(final_rows, output_path)
            print(
                f"\n[{position}/{total}] PREVIOUS FAILURE SKIPPED\n"
                f"Image: {image_path}\nSample ID: {sample_id}",
                flush=True,
            )
            continue

        if not image_object.exists():
            error = FileNotFoundError(f"Screenshot not found: {image_path}")
            record_failure(
                failure_path,
                sample_id,
                image_path,
                final_row["source"],
                adjudicator.model,
                disagreements,
                error,
            )
            final_row["groq_model"] = adjudicator.model
            final_row["adjudication_source"] = "groq_failed"
            final_row["requires_human_review"] = True
            final_row["final_annotation_notes"] = str(error)
            counters["failed"] += 1
            final_rows.append(final_row)
            save_final_rows(final_rows, output_path)
            print(
                f"\n[{position}/{total}] FAILED\n"
                f"Image: {image_path}\nError: {error}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if sample_id not in openai_records or sample_id not in gemini_records:
            missing_provider = (
                "OpenAI" if sample_id not in openai_records else "Gemini"
            )
            error = RuntimeError(
                f"{missing_provider} annotation missing for {sample_id}"
            )
            record_failure(
                failure_path,
                sample_id,
                image_path,
                final_row["source"],
                adjudicator.model,
                disagreements,
                error,
            )
            final_row["groq_model"] = adjudicator.model
            final_row["adjudication_source"] = "groq_failed"
            final_row["requires_human_review"] = True
            final_row["final_annotation_notes"] = str(error)
            counters["failed"] += 1
            final_rows.append(final_row)
            save_final_rows(final_rows, output_path)
            continue

        include_all_labels = not disagreements
        openai_annotation = filter_annotation(
            openai_records[sample_id]["annotation"],
            disagreements,
            include_all_labels,
        )
        gemini_annotation = filter_annotation(
            gemini_records[sample_id]["annotation"],
            disagreements,
            include_all_labels,
        )

        print(
            f"\n[{position}/{total}] SENDING TO GROQ\n"
            f"Image: {image_path}\n"
            f"Sample ID: {sample_id}\n"
            "Disagreement labels: "
            f"{', '.join(disagreements) or 'image quality'}\n"
            f"Model: {adjudicator.model}",
            flush=True,
        )

        try:
            result, metadata = adjudicator.annotate(
                image_path=image_path,
                openai_annotation=openai_annotation,
                gemini_annotation=gemini_annotation,
                disagreement_labels=disagreements,
            )
            result_payload = result.model_dump(mode="json")
            cache_record = {
                "sample_id": sample_id,
                "image_path": image_path,
                "source": final_row["source"],
                "disagreement_labels": disagreements,
                "annotation": result_payload,
                "metadata": metadata,
            }
            append_jsonl(groq_cache_path, cache_record)
            groq_cache[sample_id] = cache_record

            populate_groq_result(final_row, result_payload)
            final_row["groq_model"] = metadata.get(
                "model", adjudicator.model
            )
            final_row["adjudication_source"] = "groq_tiebreaker"
            final_row["requires_human_review"] = True
            final_row["final_image_quality"] = result_payload.get(
                "image_quality", ""
            )
            final_row["final_exclusion_reason"] = result_payload.get(
                "exclusion_reason", ""
            )
            final_row["final_annotation_notes"] = result_payload.get(
                "adjudication_summary",
                result_payload.get("annotation_notes", ""),
            )
            counters["new"] += 1
            print(
                f"\n[{position}/{total}] GROQ RESPONSE RECEIVED\n"
                f"Image: {image_path}",
                flush=True,
            )

        except Exception as error:
            record_failure(
                failure_path,
                sample_id,
                image_path,
                final_row["source"],
                adjudicator.model,
                disagreements,
                error,
            )
            failed_ids.add(sample_id)
            final_row["groq_model"] = adjudicator.model
            final_row["adjudication_source"] = "groq_failed"
            final_row["requires_human_review"] = True
            final_row["final_annotation_notes"] = (
                "Groq tie-breaking failed. Human review required."
            )
            counters["failed"] += 1
            print(
                f"\n[{position}/{total}] GROQ ADJUDICATION FAILED\n"
                f"Image: {image_path}\n"
                f"Sample ID: {sample_id}\n"
                f"Error type: {type(error).__name__}\n"
                f"Error: {error}\n"
                "The record was marked for human review. "
                "Processing will continue.",
                file=sys.stderr,
                flush=True,
            )

        final_rows.append(final_row)
        save_final_rows(final_rows, output_path)

        print(
            f"\n[{position}/{total}]\n"
            f"Image: {image_path}\n"
            f"Source: {final_row['adjudication_source']}\n"
            "Final labels: "
            + str(
                {
                    label: final_row[f"final_{label}"]
                    for label in LABELS
                }
            ),
            flush=True,
        )

        if arguments.sleep > 0:
            print(
                f"Waiting {arguments.sleep:.1f} seconds before "
                "the next possible Groq request.",
                flush=True,
            )
            time.sleep(arguments.sleep)

    final_frame = save_final_rows(final_rows, output_path)
    review_count = int(
        final_frame["requires_human_review"].map(as_bool).sum()
    )

    print("\n" + "=" * 76)
    print("FINAL AGREEMENT CREATED")
    print("=" * 76)
    print(f"Direct OpenAI/Gemini agreements: {counters['direct']}")
    print(f"Cached Groq decisions reused: {counters['cached']}")
    print(f"New Groq decisions: {counters['new']}")
    print(f"New Groq failures: {counters['failed']}")
    print(f"Previous failures skipped: {counters['previous_failed']}")
    print(f"Rows requiring human review: {review_count}")
    print(f"Final output: {output_path}")
    print(f"Groq cache: {groq_cache_path}")
    if counters["failed"] or counters["previous_failed"]:
        print(f"Failure log: {failure_path}")
    print(
        "\nImportant: Groq-resolved rows remain marked "
        "requires_human_review=True."
    )


if __name__ == "__main__":
    main()
