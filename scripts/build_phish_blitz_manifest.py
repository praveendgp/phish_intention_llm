#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MANIFEST_COLUMNS = [
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

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_image(
    path: Path,
) -> tuple[bool, str, int, int]:
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size

        if width < 100 or height < 100:
            return (
                False,
                f"image_too_small:{width}x{height}",
                width,
                height,
            )

        return True, "", width, height

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        return (
            False,
            f"invalid_image:{error}",
            0,
            0,
        )


def create_sample_id(
    file_hash: str,
) -> str:
    """
    Create a stable ID based on image content.

    Identical images get identical IDs, which is useful for detecting
    accidental duplicate content. Exact duplicates should already have
    been removed before running this script.
    """
    return f"phish_blitz_2025_{file_hash[:16]}"


def create_website_id(
    image_filename: str,
    file_hash: str,
) -> str:
    """
    Create a stable website identifier.

    If the flattened filename contains the original sample name, retain
    a shortened version. Otherwise, use the image content hash.
    """
    stem = Path(image_filename).stem

    if stem.startswith("phish_blitz_"):
        parts = stem.split("_")

        if len(parts) >= 4:
            candidate = "_".join(parts[3:-1]).strip("_")

            if candidate:
                return candidate[:150]

    return f"phish_blitz_site_{file_hash[:16]}"


def assign_splits(
    records: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, str]:
    """
    Reproducibly assign images to train, validation, and test.

    The split uses sample IDs instead of filesystem order.
    """
    test_ratio = 1.0 - train_ratio - validation_ratio

    if train_ratio <= 0:
        raise ValueError(
            "Train ratio must be greater than zero."
        )

    if validation_ratio < 0:
        raise ValueError(
            "Validation ratio cannot be negative."
        )

    if test_ratio <= 0:
        raise ValueError(
            "Train and validation ratios leave no test data."
        )

    sample_ids = [
        record["sample_id"]
        for record in records
    ]

    random_generator = random.Random(seed)
    random_generator.shuffle(sample_ids)

    total = len(sample_ids)

    train_count = int(total * train_ratio)
    validation_count = int(
        total * validation_ratio
    )

    train_end = train_count
    validation_end = train_count + validation_count

    split_mapping: dict[str, str] = {}

    for index, sample_id in enumerate(sample_ids):
        if index < train_end:
            split_mapping[sample_id] = "train"
        elif index < validation_end:
            split_mapping[sample_id] = "validation"
        else:
            split_mapping[sample_id] = "test"

    return split_mapping


def discover_images(
    images_directory: Path,
) -> list[Path]:
    return sorted(
        path
        for path in images_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_manifest(
    project_root: Path,
    images_directory: Path,
    output_path: Path,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> None:
    image_paths = discover_images(
        images_directory
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in:\n"
            f"{images_directory}"
        )

    records: list[dict[str, str]] = []
    hashes_seen: dict[str, Path] = {}

    invalid_count = 0
    duplicate_count = 0

    print("=" * 72)
    print("BUILDING PHISH-BLITZ MANIFEST")
    print("=" * 72)
    print(f"Images directory: {images_directory}")
    print(f"Images discovered: {len(image_paths)}")
    print()

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        file_hash = calculate_sha256(
            image_path
        )

        sample_id = create_sample_id(
            file_hash
        )

        duplicate_path = hashes_seen.get(
            file_hash
        )

        is_valid, validation_note, width, height = (
            validate_image(image_path)
        )

        if duplicate_path is not None:
            annotation_status = "excluded"
            notes = (
                f"exact_duplicate_of:"
                f"{duplicate_path.name}"
            )
            duplicate_count += 1

        elif not is_valid:
            annotation_status = "invalid"
            notes = validation_note
            invalid_count += 1
            hashes_seen[file_hash] = image_path

        else:
            annotation_status = "pending"
            notes = ""
            hashes_seen[file_hash] = image_path

        try:
            relative_image_path = (
                image_path.relative_to(
                    project_root
                )
            )
        except ValueError:
            raise RuntimeError(
                "The images directory must be inside "
                "the project root.\n"
                f"Project root: {project_root}\n"
                f"Image: {image_path}"
            )

        website_id = create_website_id(
            image_filename=image_path.name,
            file_hash=file_hash,
        )

        records.append(
            {
                "sample_id": sample_id,
                "source": "phish_blitz",
                "split": "",
                "dataset_class": "phishing",
                "phishing_status": "phishing",
                "image_path": str(
                    image_path.resolve()
                ),
                "relative_image_path": str(
                    relative_image_path
                ),
                "image_filename": image_path.name,
                "brand": "",
                "website_id": website_id,
                "screenshot_variant": "online",
                "annotation_status": (
                    annotation_status
                ),
                "annotator": "",
                "notes": notes,
                "credential_theft": "",
                "financial_fraud": "",
                "malware_distribution": "",
                "personal_information_harvesting": "",
                "_width": str(width),
                "_height": str(height),
                "_sha256": file_hash,
            }
        )

        if (
            index % 100 == 0
            or index == len(image_paths)
        ):
            print(
                f"Processed {index}/"
                f"{len(image_paths)} images"
            )

    sample_ids = [
        record["sample_id"]
        for record in records
    ]

    duplicate_sample_id_count = (
        len(sample_ids)
        - len(set(sample_ids))
    )

    if duplicate_sample_id_count:
        raise RuntimeError(
            "Duplicate sample IDs remain after image cleanup.\n"
            f"Duplicate ID count: "
            f"{duplicate_sample_id_count}\n"
            "Run the duplicate-image cleanup again before "
            "regenerating the manifest."
        )

    split_eligible_records = [
        record
        for record in records
        if record["annotation_status"] == "pending"
    ]

    split_mapping = assign_splits(
        records=split_eligible_records,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    for record in records:
        if record["sample_id"] in split_mapping:
            record["split"] = split_mapping[
                record["sample_id"]
            ]
        else:
            record["split"] = "excluded"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        ".csv.tmp"
    )

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(records)

    temporary_path.replace(output_path)

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
        "excluded": 0,
    }

    status_counts: dict[str, int] = {}

    for record in records:
        split_name = record["split"]
        status = record["annotation_status"]

        split_counts[split_name] = (
            split_counts.get(split_name, 0) + 1
        )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    print()
    print("=" * 72)
    print("MANIFEST GENERATED")
    print("=" * 72)
    print(f"Output:             {output_path}")
    print(f"Total rows:         {len(records)}")
    print(f"Unique sample IDs:  {len(set(sample_ids))}")
    print(f"Pending:            {status_counts.get('pending', 0)}")
    print(f"Invalid:            {invalid_count}")
    print(f"Duplicates found:   {duplicate_count}")
    print()
    print("Split distribution:")
    print(f"  train:            {split_counts.get('train', 0)}")
    print(
        f"  validation:       "
        f"{split_counts.get('validation', 0)}"
    )
    print(f"  test:             {split_counts.get('test', 0)}")
    print(
        f"  excluded:         "
        f"{split_counts.get('excluded', 0)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a project-compatible manifest from "
            "deduplicated Phish-Blitz screenshots."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/raw/phish_blitz/images"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/"
            "phish_blitz_manifest.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()

    if args.images.is_absolute():
        images_directory = (
            args.images.resolve()
        )
    else:
        images_directory = (
            project_root / args.images
        ).resolve()

    if args.output.is_absolute():
        output_path = args.output.resolve()
    else:
        output_path = (
            project_root / args.output
        ).resolve()

    if not project_root.exists():
        raise FileNotFoundError(
            f"Project root not found:\n"
            f"{project_root}"
        )

    if not images_directory.exists():
        raise FileNotFoundError(
            f"Images directory not found:\n"
            f"{images_directory}"
        )

    if not images_directory.is_dir():
        raise NotADirectoryError(
            f"Images path is not a directory:\n"
            f"{images_directory}"
        )

    build_manifest(
        project_root=project_root,
        images_directory=images_directory,
        output_path=output_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )


if __name__ == "__main__":
    main()