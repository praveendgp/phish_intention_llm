#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


EXISTING_PATH = Path(
    "data/processed/manifest_before_phish_blitz.csv"
)

PHISH_BLITZ_PATH = Path(
    "data/processed/phish_blitz_manifest.csv"
)

OUTPUT_PATH = Path(
    "data/processed/manifest.csv"
)

EXPECTED_COLUMNS = [
    "sample_id",
    "source",
    "split",
    "dataset_class",
    "phishing_status",
    "image_path",
    "relative_image_path",
    "image_filename",
    "brand",
    "website_id",
    "screenshot_variant",
    "annotation_status",
    "annotator",
    "notes",
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]


def normalize_column_name(column: str) -> str:
    normalized = str(column).strip()

    if (
        "data-lexical-text" in normalized
        and "brand" in normalized.lower()
    ):
        return "brand"

    if normalized.lower() == "brand":
        return "brand"

    return normalized


existing = pd.read_csv(EXISTING_PATH)
phish_blitz = pd.read_csv(PHISH_BLITZ_PATH)

existing.columns = [
    normalize_column_name(column)
    for column in existing.columns
]

if existing.columns.duplicated().any():
    duplicate_columns = existing.columns[
        existing.columns.duplicated()
    ].tolist()

    raise RuntimeError(
        f"Duplicate columns after normalization: "
        f"{duplicate_columns}"
    )

for column in EXPECTED_COLUMNS:
    if column not in existing.columns:
        existing[column] = ""

    if column not in phish_blitz.columns:
        phish_blitz[column] = ""

existing = existing[EXPECTED_COLUMNS]
phish_blitz = phish_blitz[EXPECTED_COLUMNS]

combined = pd.concat(
    [existing, phish_blitz],
    ignore_index=True,
)

duplicate_sample_ids = combined[
    combined["sample_id"].duplicated(
        keep=False
    )
]

if not duplicate_sample_ids.empty:
    print("Duplicate sample IDs:")
    print(
        duplicate_sample_ids[
            ["sample_id", "source", "image_path"]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Duplicate sample IDs found. "
        "Manifest was not written."
    )

duplicate_image_paths = combined[
    combined["image_path"].duplicated(
        keep=False
    )
]

if not duplicate_image_paths.empty:
    print("Duplicate image paths:")
    print(
        duplicate_image_paths[
            ["sample_id", "source", "image_path"]
        ].to_string(index=False)
    )

    raise RuntimeError(
        "Duplicate image paths found. "
        "Manifest was not written."
    )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

combined.to_csv(
    OUTPUT_PATH,
    index=False,
)

print(f"Previous rows:   {len(existing)}")
print(f"Phish-Blitz:     {len(phish_blitz)}")
print(f"Combined rows:   {len(combined)}")
print(f"Output:          {OUTPUT_PATH}")