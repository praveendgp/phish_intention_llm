from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )


from phishintention.annotators.consensus import (
    LABELS,
    compare,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare independent OpenAI and Gemini annotations "
            "and generate an adjudication queue."
        )
    )

    parser.add_argument(
        "--openai",
        default=(
            "data/annotations/"
            "openai_annotations.jsonl"
        ),
        help="Path to OpenAI annotations.",
    )

    parser.add_argument(
        "--gemini",
        default=(
            "data/annotations/"
            "gemini_annotations.jsonl"
        ),
        help="Path to Gemini annotations.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/annotations",
        help="Directory for agreement outputs.",
    )

    parser.add_argument(
        "--create-image-links",
        action="store_true",
        help=(
            "Create symbolic links to adjudication screenshots "
            "under adjudication_images."
        ),
    )

    return parser.parse_args()


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def create_adjudication_links(
    adjudication_frame: pd.DataFrame,
    output_directory: Path,
) -> None:
    """
    Create symbolic links to screenshots requiring adjudication.

    This avoids duplicating the underlying image files.
    """
    link_directory = (
        output_directory
        / "adjudication_images"
    )

    link_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    created_count = 0

    for queue_position, row in enumerate(
        adjudication_frame.itertuples(
            index=False
        ),
        start=1,
    ):
        source_path = Path(
            str(row.image_path)
        ).expanduser()

        if not source_path.exists():
            print(
                f"Missing adjudication image: "
                f"{source_path}",
                file=sys.stderr,
            )

            continue

        safe_name = (
            f"{queue_position:04d}_"
            f"{row.sample_id}_"
            f"{source_path.name}"
        )

        link_path = (
            link_directory
            / safe_name
        )

        if (
            link_path.exists()
            or link_path.is_symlink()
        ):
            link_path.unlink()

        link_path.symlink_to(
            source_path.resolve()
        )

        created_count += 1

    print(
        f"\nCreated {created_count} "
        f"adjudication image links under:"
    )

    print(
        link_directory
    )


def main() -> None:
    arguments = parse_arguments()

    openai_path = resolve_project_path(
        arguments.openai
    )

    gemini_path = resolve_project_path(
        arguments.gemini
    )

    output_directory = resolve_project_path(
        arguments.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_frame, metrics = compare(
        openai_path=openai_path,
        gemini_path=gemini_path,
    )

    agreement_path = (
        output_directory
        / "agreement.csv"
    )

    metrics_path = (
        output_directory
        / "agreement_metrics.json"
    )

    adjudication_path = (
        output_directory
        / "adjudication_queue.csv"
    )

    comparison_frame.to_csv(
        agreement_path,
        index=False,
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if comparison_frame.empty:
        adjudication_frame = (
            comparison_frame.copy()
        )
    else:
        adjudication_frame = (
            comparison_frame[
                comparison_frame[
                    "adjudication_required"
                ]
            ].copy()
        )

    adjudication_frame.to_csv(
        adjudication_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "ANNOTATION AGREEMENT SUMMARY"
    )

    print(
        "=" * 76
    )

    print(
        f"OpenAI annotations: "
        f"{metrics['openai_annotation_count']}"
    )

    print(
        f"Gemini annotations: "
        f"{metrics['gemini_annotation_count']}"
    )

    print(
        f"Shared annotations: "
        f"{metrics['shared_samples']}"
    )

    exact_agreement = metrics.get(
        "exact_labelset_agreement"
    )

    if exact_agreement is None:
        print(
            "Exact label-set agreement: "
            "not available"
        )
    else:
        print(
            "Exact label-set agreement: "
            f"{exact_agreement:.2%}"
        )

    print(
        "Records requiring adjudication: "
        f"{metrics['adjudication_required_count']}"
    )

    print(
        "\nPer-label agreement:"
    )

    for label in LABELS:
        label_metrics = (
            metrics["per_label"][label]
        )

        raw_agreement = (
            label_metrics["raw_agreement"]
        )

        kappa = (
            label_metrics["cohen_kappa"]
        )

        raw_text = (
            "not available"
            if raw_agreement is None
            else f"{raw_agreement:.2%}"
        )

        kappa_text = (
            "not available"
            if kappa is None
            else f"{kappa:.4f}"
        )

        print(
            f"\n- {label}"
            f"\n  Raw agreement: "
            f"{raw_text}"
            f"\n  Cohen's kappa: "
            f"{kappa_text}"
            f"\n  Disagreements: "
            f"{label_metrics['disagreement_count']}"
        )

    print(
        "\n"
        + "=" * 76
    )

    print(
        "IMAGES REQUIRING ADJUDICATION"
    )

    print(
        "=" * 76
    )

    if adjudication_frame.empty:
        print(
            "No images require adjudication."
        )
    else:
        for queue_position, row in enumerate(
            adjudication_frame.itertuples(
                index=False
            ),
            start=1,
        ):
            print(
                f"\n{queue_position}. "
                f"{row.image_path}"
            )

            print(
                f"   Sample ID: "
                f"{row.sample_id}"
            )

            disagreement_labels = str(
                row.disagreement_labels
            ).strip()

            if disagreement_labels:
                print(
                    "   Disagreement labels: "
                    f"{disagreement_labels}"
                )

            print(
                "   OpenAI quality: "
                f"{row.openai_quality}"
            )

            print(
                "   Gemini quality: "
                f"{row.gemini_quality}"
            )

    print(
        "\nFiles generated:"
    )

    print(
        f"- Complete comparison: "
        f"{agreement_path}"
    )

    print(
        f"- Agreement metrics: "
        f"{metrics_path}"
    )

    print(
        f"- Adjudication queue: "
        f"{adjudication_path}"
    )

    if (
        arguments.create_image_links
        and not adjudication_frame.empty
    ):
        create_adjudication_links(
            adjudication_frame=(
                adjudication_frame
            ),
            output_directory=(
                output_directory
            ),
        )


if __name__ == "__main__":
    main()