#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


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


def discover_images(images_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in images_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def select_keeper(paths: list[Path]) -> Path:
    """
    Keep the alphabetically first filename.

    This makes the result deterministic, meaning the same keeper
    is selected every time the script is run on the same files.
    """
    return sorted(paths, key=lambda path: path.name.lower())[0]


def find_duplicate_groups(
    image_paths: list[Path],
) -> tuple[dict[str, list[Path]], list[dict[str, str | int]]]:
    files_by_hash: dict[str, list[Path]] = {}

    total = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        file_hash = calculate_sha256(image_path)

        files_by_hash.setdefault(file_hash, []).append(
            image_path
        )

        if index % 100 == 0 or index == total:
            print(
                f"Hashed {index}/{total} images"
            )

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths in files_by_hash.items()
        if len(paths) > 1
    }

    report_rows: list[dict[str, str | int]] = []

    group_number = 0

    for file_hash, paths in sorted(
        duplicate_groups.items(),
        key=lambda item: item[0],
    ):
        group_number += 1
        keeper = select_keeper(paths)

        for path in sorted(
            paths,
            key=lambda item: item.name.lower(),
        ):
            report_rows.append(
                {
                    "duplicate_group": group_number,
                    "sha256": file_hash,
                    "action": (
                        "keep"
                        if path == keeper
                        else "delete"
                    ),
                    "filename": path.name,
                    "image_path": str(path),
                    "file_size_bytes": path.stat().st_size,
                    "keeper_filename": keeper.name,
                    "keeper_path": str(keeper),
                }
            )

    return duplicate_groups, report_rows


def write_report(
    report_path: Path,
    rows: list[dict[str, str | int]],
) -> None:
    fieldnames = [
        "duplicate_group",
        "sha256",
        "action",
        "filename",
        "image_path",
        "file_size_bytes",
        "keeper_filename",
        "keeper_path",
    ]

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find and optionally remove exact duplicate images "
            "from the flattened Phish-Blitz image directory."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/raw/phish_blitz/images"
        ),
        help="Directory containing flattened screenshot images.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/processed/"
            "phish_blitz_duplicate_images.csv"
        ),
        help="CSV report containing keeper and duplicate decisions.",
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help=(
            "Permanently delete exact duplicate copies. "
            "Without this option, the script performs a dry run."
        ),
    )

    args = parser.parse_args()

    images_directory = args.images.resolve()
    report_path = args.report.resolve()

    if not images_directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {images_directory}"
        )

    if not images_directory.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {images_directory}"
        )

    image_paths = discover_images(images_directory)

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in: "
            f"{images_directory}"
        )

    print("=" * 72)
    print("EXACT DUPLICATE IMAGE CHECK")
    print("=" * 72)
    print(f"Image directory: {images_directory}")
    print(f"Images found:    {len(image_paths)}")
    print(
        f"Mode:            "
        f"{'DELETE' if args.delete else 'DRY RUN'}"
    )
    print()

    duplicate_groups, report_rows = find_duplicate_groups(
        image_paths
    )

    rows_to_delete = [
        row
        for row in report_rows
        if row["action"] == "delete"
    ]

    duplicate_bytes = sum(
        int(row["file_size_bytes"])
        for row in rows_to_delete
    )

    write_report(
        report_path=report_path,
        rows=report_rows,
    )

    print()
    print("=" * 72)
    print("DUPLICATE SUMMARY" * 72)
    print(f"Total images:          {len(image_paths)}")
    print(
        f"Duplicate groups:      "
        f"{len(duplicate_groups)}"
    )
    print(
        f"Duplicate files:       "
        f"{len(rows_to_delete)}"
    )
    print(
        f"Space recoverable:     "
        f"{duplicate_bytes / (1024 * 1024):.2f} MB"
    )
    print(f"Report:                {report_path}")
    print()

    if not rows_to_delete:
        print("No exact duplicate images were found.")
        return

    print("First 20 duplicate files selected for deletion:")
    print("-" * 72)

    for row in rows_to_delete[:20]:
        print(f"DELETE: {row['filename']}")
        print(f"KEEP:   {row['keeper_filename']}")
        print(f"HASH:   {row['sha256']}")
        print()

    if len(rows_to_delete) > 20:
        print(
            f"...and {len(rows_to_delete) - 20} "
            f"additional duplicate files."
        )
        print()

    if not args.delete:
        print("=" * 72)
        print("DRY RUN COMPLETE")
        print("=" * 72)
        print("No images were deleted.")
        print()
        print("Review the report before deletion:")
        print(f"  {report_path}")
        print()
        print("To delete the reported exact duplicates, rerun")
        print("this script with the --delete option.")
        return

    deleted_count = 0
    deleted_bytes = 0

    print("=" * 72)
    print("DELETING EXACT DUPLICATES")
    print("=" * 72)

    for row in rows_to_delete:
        duplicate_path = Path(
            str(row["image_path"])
        )

        keeper_path = Path(
            str(row["keeper_path"])
        )

        if not keeper_path.exists():
            raise RuntimeError(
                "Refusing deletion because the keeper is missing:\n"
                f"Keeper:    {keeper_path}\n"
                f"Duplicate: {duplicate_path}"
            )

        if not duplicate_path.exists():
            print(
                f"Already missing, skipped: "
                f"{duplicate_path.name}"
            )
            continue

        duplicate_hash = calculate_sha256(
            duplicate_path
        )

        keeper_hash = calculate_sha256(
            keeper_path
        )

        expected_hash = str(row["sha256"])

        if (
            duplicate_hash != expected_hash
            or keeper_hash != expected_hash
        ):
            raise RuntimeError(
                "Hash changed after report generation. "
                "Deletion stopped.\n"
                f"Duplicate: {duplicate_path}\n"
                f"Keeper: {keeper_path}"
            )

        file_size = duplicate_path.stat().st_size

        duplicate_path.unlink()

        if duplicate_path.exists():
            raise RuntimeError(
                f"Could not delete: {duplicate_path}"
            )

        deleted_count += 1
        deleted_bytes += file_size

        if (
            deleted_count % 100 == 0
            or deleted_count == len(rows_to_delete)
        ):
            print(
                f"Deleted {deleted_count}/"
                f"{len(rows_to_delete)} duplicates"
            )

    remaining_images = discover_images(
        images_directory
    )

    remaining_hashes: set[str] = set()
    remaining_duplicate_count = 0

    print()
    print("Verifying the cleaned directory...")

    for index, image_path in enumerate(
        remaining_images,
        start=1,
    ):
        file_hash = calculate_sha256(image_path)

        if file_hash in remaining_hashes:
            remaining_duplicate_count += 1
        else:
            remaining_hashes.add(file_hash)

        if (
            index % 100 == 0
            or index == len(remaining_images)
        ):
            print(
                f"Verified {index}/"
                f"{len(remaining_images)} images"
            )

    print()
    print("=" * 72)
    print("CLEANUP COMPLETE")
    print("=" * 72)
    print(f"Images before:       {len(image_paths)}")
    print(f"Duplicates deleted:  {deleted_count}")
    print(f"Images remaining:    {len(remaining_images)}")
    print(
        f"Space recovered:     "
        f"{deleted_bytes / (1024 * 1024):.2f} MB"
    )
    print(
        f"Remaining exact "
        f"duplicates:          "
        f"{remaining_duplicate_count}"
    )

    if remaining_duplicate_count:
        raise RuntimeError(
            "Verification detected remaining exact duplicates."
        )


if __name__ == "__main__":
    main()