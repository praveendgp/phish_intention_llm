from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
)


LABELS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]


def load_jsonl(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load provider annotations and index them by sample ID.
    """
    annotation_path = Path(path)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: "
            f"{annotation_path}"
        )

    annotations = {}

    for line_number, line in enumerate(
        annotation_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in "
                f"{annotation_path} "
                f"at line {line_number}: "
                f"{error}"
            ) from error

        sample_id = item.get(
            "sample_id"
        )

        if not sample_id:
            raise ValueError(
                f"Missing sample_id in "
                f"{annotation_path} "
                f"at line {line_number}"
            )

        annotations[
            str(sample_id)
        ] = item

    return annotations


def get_label_decision(
    record: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """
    Return one label decision from an annotation record.
    """
    annotation = record.get(
        "annotation",
        {},
    )

    decision = annotation.get(
        label,
        {},
    )

    if not isinstance(
        decision,
        dict,
    ):
        decision = {}

    return decision


def get_binary_label(
    record: dict[str, Any],
    label: str,
) -> int:
    """
    Convert one provider decision to 0 or 1.
    """
    decision = get_label_decision(
        record,
        label,
    )

    return int(
        bool(
            decision.get(
                "present",
                False,
            )
        )
    )


def get_confidence(
    record: dict[str, Any],
    label: str,
) -> float | None:
    """
    Return one provider confidence value.
    """
    decision = get_label_decision(
        record,
        label,
    )

    value = decision.get(
        "confidence"
    )

    if value is None:
        return None

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def get_evidence_text(
    record: dict[str, Any],
    label: str,
) -> str:
    """
    Combine evidence statements into one CSV-friendly string.
    """
    decision = get_label_decision(
        record,
        label,
    )

    evidence = decision.get(
        "evidence",
        [],
    )

    if isinstance(evidence, str):
        evidence = [evidence]

    if not isinstance(
        evidence,
        list,
    ):
        return ""

    cleaned_evidence = []

    for item in evidence:
        text = str(item).strip()

        if text:
            cleaned_evidence.append(
                text
            )

    return " | ".join(
        cleaned_evidence
    )


def get_image_quality(
    record: dict[str, Any],
) -> str:
    annotation = record.get(
        "annotation",
        {},
    )

    return str(
        annotation.get(
            "image_quality",
            "",
        )
    ).strip()


def get_annotation_notes(
    record: dict[str, Any],
) -> str:
    annotation = record.get(
        "annotation",
        {},
    )

    return str(
        annotation.get(
            "annotation_notes",
            "",
        )
    ).strip()


def get_model(
    record: dict[str, Any],
) -> str:
    metadata = record.get(
        "metadata",
        {},
    )

    return str(
        metadata.get(
            "model",
            "",
        )
    ).strip()


def positive_agreement(
    left: list[int],
    right: list[int],
) -> float | None:
    """
    Calculate positive agreement:

        2a / (2a + b + c)

    where:
        a = both positive
        b = left positive, right negative
        c = left negative, right positive
    """
    both_positive = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 1
        and right_value == 1
    )

    left_only = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 1
        and right_value == 0
    )

    right_only = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 0
        and right_value == 1
    )

    denominator = (
        2 * both_positive
        + left_only
        + right_only
    )

    if denominator == 0:
        return None

    return (
        2 * both_positive
        / denominator
    )


def negative_agreement(
    left: list[int],
    right: list[int],
) -> float | None:
    """
    Calculate negative agreement:

        2d / (2d + b + c)

    where:
        d = both negative
        b = left positive, right negative
        c = left negative, right positive
    """
    both_negative = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 0
        and right_value == 0
    )

    left_only = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 1
        and right_value == 0
    )

    right_only = sum(
        1
        for left_value, right_value
        in zip(left, right)
        if left_value == 0
        and right_value == 1
    )

    denominator = (
        2 * both_negative
        + left_only
        + right_only
    )

    if denominator == 0:
        return None

    return (
        2 * both_negative
        / denominator
    )


def calculate_kappa(
    left: list[int],
    right: list[int],
) -> float | None:
    """
    Calculate Cohen's kappa when sufficient class variation exists.
    """
    combined_values = set(
        left + right
    )

    if len(combined_values) < 2:
        return None

    return float(
        cohen_kappa_score(
            left,
            right,
        )
    )


def compare(
    openai_path: str | Path,
    gemini_path: str | Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """
    Compare independent OpenAI and Gemini annotations.

    The returned DataFrame contains image paths, predictions,
    confidence values, evidence, agreement fields, and an
    adjudication-required flag.
    """
    openai_annotations = load_jsonl(
        openai_path
    )

    gemini_annotations = load_jsonl(
        gemini_path
    )

    shared_sample_ids = sorted(
        set(openai_annotations)
        & set(gemini_annotations)
    )

    comparison_rows = []

    for sample_id in shared_sample_ids:
        openai_record = (
            openai_annotations[
                sample_id
            ]
        )

        gemini_record = (
            gemini_annotations[
                sample_id
            ]
        )

        openai_image_path = str(
            openai_record.get(
                "image_path",
                "",
            )
        ).strip()

        gemini_image_path = str(
            gemini_record.get(
                "image_path",
                "",
            )
        ).strip()

        # Prefer the OpenAI path, falling back to Gemini.
        image_path = (
            openai_image_path
            or gemini_image_path
        )

        openai_quality = (
            get_image_quality(
                openai_record
            )
        )

        gemini_quality = (
            get_image_quality(
                gemini_record
            )
        )

        row = {
            "sample_id": sample_id,
            "image_path": image_path,
            "source": (
                openai_record.get(
                    "source"
                )
                or gemini_record.get(
                    "source"
                )
                or ""
            ),
            "openai_model": get_model(
                openai_record
            ),
            "gemini_model": get_model(
                gemini_record
            ),
            "openai_quality": (
                openai_quality
            ),
            "gemini_quality": (
                gemini_quality
            ),
            "openai_notes": (
                get_annotation_notes(
                    openai_record
                )
            ),
            "gemini_notes": (
                get_annotation_notes(
                    gemini_record
                )
            ),
        }

        all_labels_agree = True

        disagreement_labels = []

        for label in LABELS:
            openai_value = (
                get_binary_label(
                    openai_record,
                    label,
                )
            )

            gemini_value = (
                get_binary_label(
                    gemini_record,
                    label,
                )
            )

            label_agrees = (
                openai_value
                == gemini_value
            )

            row[
                f"openai_{label}"
            ] = openai_value

            row[
                f"gemini_{label}"
            ] = gemini_value

            row[
                f"agree_{label}"
            ] = label_agrees

            row[
                f"openai_{label}"
                "_confidence"
            ] = get_confidence(
                openai_record,
                label,
            )

            row[
                f"gemini_{label}"
                "_confidence"
            ] = get_confidence(
                gemini_record,
                label,
            )

            row[
                f"openai_{label}"
                "_evidence"
            ] = get_evidence_text(
                openai_record,
                label,
            )

            row[
                f"gemini_{label}"
                "_evidence"
            ] = get_evidence_text(
                gemini_record,
                label,
            )

            if not label_agrees:
                all_labels_agree = False

                disagreement_labels.append(
                    label
                )

        quality_agrees = (
            openai_quality
            == gemini_quality
        )

        both_usable = (
            openai_quality == "usable"
            and gemini_quality == "usable"
        )

        row[
            "exact_labelset_agreement"
        ] = all_labels_agree

        row[
            "quality_agreement"
        ] = quality_agrees

        row[
            "disagreement_labels"
        ] = " | ".join(
            disagreement_labels
        )

        row[
            "adjudication_required"
        ] = (
            not all_labels_agree
            or not quality_agrees
            or not both_usable
        )

        # Empty fields for the human reviewer to complete.
        row[
            "final_credential_theft"
        ] = ""

        row[
            "final_financial_fraud"
        ] = ""

        row[
            "final_malware_distribution"
        ] = ""

        row[
            "final_personal_information_harvesting"
        ] = ""

        row[
            "adjudication_notes"
        ] = ""

        comparison_rows.append(
            row
        )

    comparison_frame = pd.DataFrame(
        comparison_rows
    )

    metrics: dict[str, Any] = {
        "openai_annotation_count": len(
            openai_annotations
        ),
        "gemini_annotation_count": len(
            gemini_annotations
        ),
        "shared_samples": len(
            comparison_frame
        ),
        "missing_from_openai": sorted(
            set(gemini_annotations)
            - set(openai_annotations)
        ),
        "missing_from_gemini": sorted(
            set(openai_annotations)
            - set(gemini_annotations)
        ),
        "exact_labelset_agreement": None,
        "adjudication_required_count": 0,
        "per_label": {},
    }

    if comparison_frame.empty:
        for label in LABELS:
            metrics[
                "per_label"
            ][label] = {
                "raw_agreement": None,
                "cohen_kappa": None,
                "positive_agreement": None,
                "negative_agreement": None,
                "disagreement_count": 0,
            }

        return (
            comparison_frame,
            metrics,
        )

    metrics[
        "exact_labelset_agreement"
    ] = float(
        comparison_frame[
            "exact_labelset_agreement"
        ].mean()
    )

    metrics[
        "adjudication_required_count"
    ] = int(
        comparison_frame[
            "adjudication_required"
        ].sum()
    )

    for label in LABELS:
        openai_values = (
            comparison_frame[
                f"openai_{label}"
            ]
            .astype(int)
            .tolist()
        )

        gemini_values = (
            comparison_frame[
                f"gemini_{label}"
            ]
            .astype(int)
            .tolist()
        )

        raw_agreement = float(
            comparison_frame[
                f"agree_{label}"
            ].mean()
        )

        disagreement_count = int(
            (
                ~comparison_frame[
                    f"agree_{label}"
                ]
            ).sum()
        )

        metrics[
            "per_label"
        ][label] = {
            "raw_agreement": (
                raw_agreement
            ),
            "cohen_kappa": (
                calculate_kappa(
                    openai_values,
                    gemini_values,
                )
            ),
            "positive_agreement": (
                positive_agreement(
                    openai_values,
                    gemini_values,
                )
            ),
            "negative_agreement": (
                negative_agreement(
                    openai_values,
                    gemini_values,
                )
            ),
            "disagreement_count": (
                disagreement_count
            ),
            "openai_positive_count": (
                int(
                    sum(
                        openai_values
                    )
                )
            ),
            "gemini_positive_count": (
                int(
                    sum(
                        gemini_values
                    )
                )
            ),
        }

    return (
        comparison_frame,
        metrics,
    )