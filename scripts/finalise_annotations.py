from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


LABELS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]


def as_boolean(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return (
            not pd.isna(value)
            and value != 0
        )

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
        }
    )


def as_binary(
    value: Any,
) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        if pd.isna(value):
            raise ValueError(
                "Label is empty."
            )

        if value in {0, 1}:
            return int(value)

    text = (
        str(value)
        .strip()
        .lower()
    )

    if text in {
        "1",
        "true",
        "yes",
        "y",
    }:
        return 1

    if text in {
        "0",
        "false",
        "no",
        "n",
    }:
        return 0

    raise ValueError(
        f"Invalid binary label: {value!r}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create final adjudicated ground truth "
            "from final agreement and human review."
        )
    )

    parser.add_argument(
        "--agreement",
        default=(
            "data/annotations/"
            "final_agreement.csv"
        ),
    )

    parser.add_argument(
        "--review",
        default=(
            "data/annotations/"
            "human_review_queue.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/annotations/"
            "final_adjudicated.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    agreement_path = Path(
        arguments.agreement
    )

    review_path = Path(
        arguments.review
    )

    output_path = Path(
        arguments.output
    )

    if not agreement_path.exists():
        raise SystemExit(
            f"Agreement file not found: "
            f"{agreement_path}"
        )

    agreement = pd.read_csv(
        agreement_path
    ).fillna("")

    if review_path.exists():
        review = pd.read_csv(
            review_path
        ).fillna("")

        review = review.drop_duplicates(
            subset=["sample_id"],
            keep="last",
        )

        review = review.set_index(
            "sample_id"
        )

    else:
        review = pd.DataFrame()

    final_rows = []
    incomplete_rows = []

    for agreement_row in (
        agreement.itertuples(
            index=False
        )
    ):
        sample_id = str(
            agreement_row.sample_id
        )

        needs_review = as_boolean(
            agreement_row
            .requires_human_review
        )

        selected_values = {
            label: getattr(
                agreement_row,
                f"final_{label}",
                "",
            )
            for label in LABELS
        }

        annotation_source = (
            "openai_gemini_agreement"
        )

        adjudication_notes = str(
            getattr(
                agreement_row,
                "final_annotation_notes",
                "",
            )
        ).strip()

        if needs_review:
            if review.empty:
                incomplete_rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": getattr(
                            agreement_row,
                            "image_path",
                            "",
                        ),
                        "reason": (
                            "Human review file "
                            "does not exist."
                        ),
                    }
                )

                continue

            if sample_id not in review.index:
                incomplete_rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": getattr(
                            agreement_row,
                            "image_path",
                            "",
                        ),
                        "reason": (
                            "Record is missing "
                            "from review queue."
                        ),
                    }
                )

                continue

            review_row = review.loc[
                sample_id
            ]

            if isinstance(
                review_row,
                pd.DataFrame,
            ):
                review_row = (
                    review_row.iloc[-1]
                )

            review_completed = (
                as_boolean(
                    review_row.get(
                        "human_review_completed",
                        "",
                    )
                )
            )

            if not review_completed:
                incomplete_rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": getattr(
                            agreement_row,
                            "image_path",
                            "",
                        ),
                        "reason": (
                            "Human review is "
                            "not completed."
                        ),
                    }
                )

                continue

            selected_values = {
                label: review_row.get(
                    f"final_{label}",
                    "",
                )
                for label in LABELS
            }

            annotation_source = (
                "human_review"
            )

            adjudication_notes = str(
                review_row.get(
                    "human_review_notes",
                    "",
                )
            ).strip()

        final_record = {
            "sample_id": sample_id,
            "image_path": str(
                getattr(
                    agreement_row,
                    "image_path",
                    "",
                )
            ),
            "source": str(
                getattr(
                    agreement_row,
                    "source",
                    "",
                )
            ),
            "annotation_status": (
                "adjudicated"
            ),
            "annotation_source": (
                annotation_source
            ),
            "original_decision_source": str(
                getattr(
                    agreement_row,
                    "adjudication_source",
                    "",
                )
            ),
            "adjudication_notes": (
                adjudication_notes
            ),
        }

        invalid_label = False

        for label in LABELS:
            try:
                final_record[label] = (
                    as_binary(
                        selected_values[
                            label
                        ]
                    )
                )

            except ValueError as error:
                incomplete_rows.append(
                    {
                        "sample_id": (
                            sample_id
                        ),
                        "image_path": (
                            final_record[
                                "image_path"
                            ]
                        ),
                        "reason": (
                            f"{label}: {error}"
                        ),
                    }
                )

                invalid_label = True
                break

        if not invalid_label:
            final_rows.append(
                final_record
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if incomplete_rows:
        incomplete_path = (
            output_path.parent
            / "incomplete_adjudication.csv"
        )

        pd.DataFrame(
            incomplete_rows
        ).to_csv(
            incomplete_path,
            index=False,
        )

        raise SystemExit(
            "Finalisation stopped because "
            "some records are incomplete. "
            f"Review: {incomplete_path}"
        )

    final_frame = pd.DataFrame(
        final_rows
    )

    final_frame.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Created: {output_path}"
    )

    print(
        f"Adjudicated records: "
        f"{len(final_frame)}"
    )

    print(
        "\nAnnotation sources:"
    )

    print(
        final_frame[
            "annotation_source"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nFinal label counts:"
    )

    for label in LABELS:
        print(
            f"{label}: "
            f"{int(final_frame[label].sum())}"
        )


if __name__ == "__main__":
    main()