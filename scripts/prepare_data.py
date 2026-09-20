from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
SPLITS = ("train", "val", "test")
CLASSES = ("legitimate", "phishing")
INTENTS = (
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest.csv from image/{train,val,test}/{legitimate,phishing}."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root containing the image directory.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/manifest.csv",
    )
    parser.add_argument(
        "--source-name",
        default="phishing_dataset",
    )
    return parser.parse_args()


def stable_sample_id(source_name: str, relative_path: Path) -> str:
    key = f"{source_name}:{relative_path.as_posix()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def validate_layout(dataset_root: Path) -> Path:
    image_root = dataset_root / "image"
    missing = []
    for split in SPLITS:
        for class_name in CLASSES:
            folder = image_root / split / class_name
            if not folder.is_dir():
                missing.append(str(folder))
    if missing:
        raise SystemExit(
            "Dataset layout is incomplete. Missing folders:\n- "
            + "\n- ".join(missing)
        )
    return image_root


def build_manifest(dataset_root: Path, source_name: str) -> pd.DataFrame:
    image_root = validate_layout(dataset_root)
    rows = []

    for split in SPLITS:
        for class_name in CLASSES:
            folder = image_root / split / class_name
            paths = sorted(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in paths:
                relative_path = image_path.relative_to(dataset_root)
                is_phishing = class_name == "phishing"
                row = {
                    "sample_id": stable_sample_id(source_name, relative_path),
                    "source": source_name,
                    "split": split,
                    "dataset_class": class_name,
                    "phishing_status": class_name,
                    "image_path": str(image_path.resolve()),
                    "relative_image_path": relative_path.as_posix(),
                    "image_filename": image_path.name,
                    "brand": "",
                    "website_id": "",
                    "screenshot_variant": "",
                    "annotation_status": "unlabelled" if is_phishing else "not_applicable",
                    "annotator": "",
                    "notes": "",
                }
                for intent in INTENTS:
                    row[intent] = "" if is_phishing else 0
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No supported image files were found.")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(False), "sample_id"]
        raise SystemExit(f"Duplicate sample IDs detected: {duplicates.tolist()}")
    return frame


def main() -> None:
    args = parse_arguments()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    frame = build_manifest(dataset_root, args.source_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Created: {output}")
    print(f"Total images: {len(frame)}")
    print("\nCounts by split and class:")
    print(frame.groupby(["split", "dataset_class"]).size().to_string())
    print("\nIntention-annotation candidates:", int((frame["phishing_status"] == "phishing").sum()))
    print("Legitimate reference images:", int((frame["phishing_status"] == "legitimate").sum()))


if __name__ == "__main__":
    main()
