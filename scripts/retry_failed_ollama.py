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

load_dotenv(PROJECT_ROOT / ".env")

from phishintention.annotators import OllamaAdjudicator
from phishintention.annotators.consensus import LABELS, load_jsonl

FAILED_SOURCES = {
    "groq_failed",
    "groq_failed_previous_run",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry only Groq-failed adjudication rows and update "
            "final_agreement.csv in place."
        )
    )
    parser.add_argument(
        "--agreement",
        default="data/annotations/final_agreement.csv",
        help="Combined agreement CSV containing Groq failure rows.",
    )
    parser.add_argument(
        "--openai",
        default="data/annotations/openai_annotations.jsonl",
        help="OpenAI annotation JSONL.",
    )
    parser.add_argument(
        "--gemini",
        default="data/annotations/gemini_annotations.jsonl",
        help="Gemini annotation JSONL.",
    )
    parser.add_argument(
        "--groq-cache",
        default="data/annotations/ollama_adjudications.jsonl",
        help="Successful Groq adjudication cache.",
    )
    parser.add_argument(
        "--failure-log",
        default="data/annotations/ollama_adjudication_failures.jsonl",
        help="Groq failure JSONL. Successful retries are removed from it.",
    )
    parser.add_argument(
        "--human-review-queue",
        default="data/annotations/human_review_queue.csv",
        help="Existing human-review queue to update when present.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=20.0,
        help="Delay between API calls to reduce rate-limit errors.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Retry only a particular sample ID. Repeat for multiple IDs.",
    )
    parser.add_argument(
        "--include-empty-source",
        action="store_true",
        help="Also retry rows with empty adjudication_source and empty final labels.",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            print(
                f"Warning: skipped invalid JSON in {path}, "
                f"line {line_number}: {error}",
                file=sys.stderr,
            )
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def rewrite_failure_log(path: Path, successful_ids: set[str]) -> None:
    if not path.exists() or not successful_ids:
        return

    remaining = [
        record
        for record in read_jsonl(path)
        if str(record.get("sample_id", "")) not in successful_ids
    ]

    with path.open("w", encoding="utf-8") as handle:
        for record in remaining:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_disagreement_labels(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []

    text = text.replace(",", "|")
    labels = [item.strip() for item in text.split("|") if item.strip()]
    return [label for label in labels if label in LABELS]


def filter_annotation(
    annotation: dict[str, Any],
    disagreement_labels: list[str],
) -> dict[str, Any]:
    selected = disagreement_labels or list(LABELS)
    filtered = {
        label: annotation[label]
        for label in selected
        if label in annotation
    }
    for field in ("image_quality", "exclusion_reason", "annotation_notes"):
        filtered[field] = annotation.get(field, "")
    return filtered


def set_result_columns(
    frame: pd.DataFrame,
    index: int,
    annotation: dict[str, Any],
    model: str,
) -> None:
    """
    Store a successful Groq retry in final_agreement.csv.

    All values are written as strings because the CSV is loaded
    using dtype=str.
    """
    for label in LABELS:
        decision = annotation.get(
            label,
            {},
        )

        if not isinstance(
            decision,
            dict,
        ):
            decision = {}

        evidence = decision.get(
            "evidence",
            [],
        )

        if isinstance(
            evidence,
            str,
        ):
            evidence = [evidence]

        if not isinstance(
            evidence,
            list,
        ):
            evidence = []

        binary_value = str(
            int(
                bool(
                    decision.get(
                        "present",
                        False,
                    )
                )
            )
        )

        confidence = decision.get(
            "confidence",
            "",
        )

        confidence_text = (
            ""
            if confidence is None
            else str(confidence)
        )

        evidence_text = " | ".join(
            str(item).strip()
            for item in evidence
            if str(item).strip()
        )

        frame.at[
            index,
            f"final_{label}",
        ] = binary_value

        frame.at[
            index,
            f"final_{label}_confidence",
        ] = confidence_text

        frame.at[
            index,
            f"final_{label}_evidence",
        ] = evidence_text

    frame.at[
        index,
        "groq_model",
    ] = str(model)

    frame.at[
        index,
        "adjudication_source",
    ] = "ollama_tiebreaker_retry"

    frame.at[
        index,
        "requires_human_review",
    ] = "True"

    frame.at[
        index,
        "final_image_quality",
    ] = str(
        annotation.get(
            "image_quality",
            "",
        )
    )

    frame.at[
        index,
        "final_exclusion_reason",
    ] = str(
        annotation.get(
            "exclusion_reason",
            "",
        )
    )

    frame.at[
        index,
        "final_annotation_notes",
    ] = str(
        annotation.get(
            "adjudication_summary",
            annotation.get(
                "annotation_notes",
                "",
            ),
        )
    )


def update_human_review_queue(
    queue_path: Path,
    agreement: pd.DataFrame,
    successful_ids: set[str],
) -> None:
    if not queue_path.exists() or not successful_ids:
        return

    queue = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
    agreement_by_id = agreement.set_index("sample_id", drop=False)

    for queue_index, queue_row in queue.iterrows():
        sample_id = str(queue_row.get("sample_id", ""))
        if sample_id not in successful_ids or sample_id not in agreement_by_id.index:
            continue

        source_row = agreement_by_id.loc[sample_id]
        if isinstance(source_row, pd.DataFrame):
            source_row = source_row.iloc[-1]

        for column in agreement.columns:
            if column in queue.columns:
                queue.at[queue_index, column] = source_row[column]

        # The machine retry succeeded, but the row still needs human approval.
        if "human_review_completed" in queue.columns:
            queue.at[queue_index, "human_review_completed"] = ""
        if "human_review_notes" in queue.columns:
            queue.at[queue_index, "human_review_notes"] = ""

    queue.to_csv(queue_path, index=False)


def main() -> None:
    args = parse_arguments()

    agreement_path = resolve_path(args.agreement)
    openai_path = resolve_path(args.openai)
    gemini_path = resolve_path(args.gemini)
    cache_path = resolve_path(args.groq_cache)
    failure_path = resolve_path(args.failure_log)
    review_queue_path = resolve_path(args.human_review_queue)

    if not agreement_path.exists():
        raise SystemExit(f"Agreement file not found: {agreement_path}")
    if not openai_path.exists():
        raise SystemExit(f"OpenAI annotations not found: {openai_path}")
    if not gemini_path.exists():
        raise SystemExit(f"Gemini annotations not found: {gemini_path}")

    agreement = pd.read_csv(
        agreement_path,
        dtype=str,
        keep_default_na=False,
    )

    required = {"sample_id", "image_path", "adjudication_source"}
    missing = required - set(agreement.columns)
    if missing:
        raise SystemExit(
            "final_agreement.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    source_mask = agreement["adjudication_source"].isin(FAILED_SOURCES)

    if args.include_empty_source:
        final_columns = [f"final_{label}" for label in LABELS]
        empty_final = agreement[final_columns].apply(
            lambda row: all(not str(value).strip() for value in row),
            axis=1,
        )
        source_mask = source_mask | (
            agreement["adjudication_source"].str.strip().eq("") & empty_final
        )

    if args.sample_id:
        requested = {str(item) for item in args.sample_id}
        source_mask = source_mask & agreement["sample_id"].isin(requested)

    failed_rows = agreement[source_mask].copy()
    if args.limit is not None:
        failed_rows = failed_rows.head(args.limit)

    print(f"Failed rows selected for retry: {len(failed_rows)}")
    if failed_rows.empty:
        print("Nothing to retry.")
        return

    openai_records = load_jsonl(openai_path)
    gemini_records = load_jsonl(gemini_path)
    adjudicator = OllamaAdjudicator(
    model=args.model
)

    successful_ids: set[str] = set()
    retry_failures: list[dict[str, Any]] = []

    for position, (index, row) in enumerate(failed_rows.iterrows(), start=1):
        sample_id = str(row["sample_id"])
        image_path = Path(str(row["image_path"])).expanduser()
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        image_path = image_path.resolve()
        disagreements = parse_disagreement_labels(
            row.get("disagreement_labels", "")
        )

        print(
            f"\n[{position}/{len(failed_rows)}] RETRYING GROQ FAILURE"
            f"\nImage: {image_path}"
            f"\nSample ID: {sample_id}"
            f"\nDisagreement labels: "
            f"{', '.join(disagreements) or 'all labels / image quality'}"
            f"\nModel: {adjudicator.model}",
            flush=True,
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            if sample_id not in openai_records:
                raise KeyError(f"OpenAI annotation missing for {sample_id}")
            if sample_id not in gemini_records:
                raise KeyError(f"Gemini annotation missing for {sample_id}")

            openai_annotation = filter_annotation(
                openai_records[sample_id]["annotation"],
                disagreements,
            )
            gemini_annotation = filter_annotation(
                gemini_records[sample_id]["annotation"],
                disagreements,
            )

            result, metadata = adjudicator.annotate(
                image_path=image_path,
                openai_annotation=openai_annotation,
                gemini_annotation=gemini_annotation,
                disagreement_labels=disagreements,
            )
            payload = result.model_dump(mode="json")

            append_jsonl(
                cache_path,
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "source": row.get("source", ""),
                    "disagreement_labels": disagreements,
                    "annotation": payload,
                    "metadata": {
                        **metadata,
                        "retry_only_script": True,
                    },
                },
            )

            set_result_columns(
                agreement,
                index,
                payload,
                metadata.get("model", adjudicator.model),
            )
            agreement.to_csv(agreement_path, index=False)
            successful_ids.add(sample_id)

            print(
                "Retry succeeded."
                f"\nFinal labels: "
                + str(
                    {
                        label: agreement.at[index, f"final_{label}"]
                        for label in LABELS
                    }
                ),
                flush=True,
            )

        except Exception as error:
            failure_record = {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "source": row.get("source", ""),
                "model": adjudicator.model,
                "disagreement_labels": disagreements,
                "error_type": type(error).__name__,
                "error": str(error),
                "retry_only_script": True,
            }
            retry_failures.append(failure_record)
            append_jsonl(failure_path, failure_record)

            print(
                "Retry failed."
                f"\nError type: {type(error).__name__}"
                f"\nError: {error}",
                file=sys.stderr,
                flush=True,
            )

        if args.sleep > 0 and position < len(failed_rows):
            print(f"Waiting {args.sleep:.1f} seconds...", flush=True)
            time.sleep(args.sleep)

    # Remove all old failure-log entries for successful sample IDs.
    rewrite_failure_log(failure_path, successful_ids)

    # Reflect successful retries in an already-created human review queue.
    update_human_review_queue(
        review_queue_path,
        agreement,
        successful_ids,
    )

    print("\n" + "=" * 72)
    print("FAILED-ONLY GROQ RETRY COMPLETE")
    print("=" * 72)
    print(f"Selected: {len(failed_rows)}")
    print(f"Succeeded: {len(successful_ids)}")
    print(f"Still failed: {len(retry_failures)}")
    print(f"Updated agreement: {agreement_path}")
    print(f"Groq cache: {cache_path}")
    print(f"Failure log: {failure_path}")
    if review_queue_path.exists():
        print(f"Updated human review queue: {review_queue_path}")
    print(
        "Successful retries remain requires_human_review=True; "
        "review them before finalising ground truth."
    )


if __name__ == "__main__":
    main()
