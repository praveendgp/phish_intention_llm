#!/usr/bin/env python3
"""Migrate PhishIntentionLLM to a split/class-folder screenshot dataset.

Expected source layout:

    <dataset-root>/
    ├── image/
    │   ├── train/
    │   │   ├── legitimate/
    │   │   └── phishing/
    │   ├── val/
    │   │   ├── legitimate/
    │   │   └── phishing/
    │   └── test/
    │       ├── legitimate/
    │       └── phishing/
    └── url/                    # optional; preserved but not used by the VLM pipeline

The migration is intentionally conservative:
- Existing project files are backed up before replacement.
- Old annotation artefacts are archived because sample IDs change.
- Old raw datasets are removed only with --remove-old-raw.
- Existing outputs are archived only with --archive-outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
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

PREPARE_DATA_SOURCE = r'''from __future__ import annotations

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
'''

README_SECTION = r'''
<!-- NEW_DATASET_WORKFLOW_START -->
## Dataset layout

The project uses one screenshot dataset with the following structure:

```text
<dataset-root>/
├── image/
│   ├── train/
│   │   ├── legitimate/
│   │   └── phishing/
│   ├── val/
│   │   ├── legitimate/
│   │   └── phishing/
│   └── test/
│       ├── legitimate/
│       └── phishing/
└── url/                         # Optional metadata; not used for screenshot inference
```

Generate the manifest:

```bash
python scripts/prepare_data.py \
  --dataset-root "/absolute/path/to/Phishing dataset" \
  --output data/processed/manifest.csv \
  --source-name phishing_dataset
```

The manifest includes both classes, but only rows with:

```text
phishing_status = phishing
annotation_status = unlabelled
```

are sent through the four-intention annotation workflow. Legitimate screenshots are retained as reference or phishing-detection data and receive:

```text
annotation_status = not_applicable
```

because the four phishing-intention labels do not apply to legitimate screenshots.

Start the Streamlit application:

```bash
python -m streamlit run app.py
```
<!-- NEW_DATASET_WORKFLOW_END -->
'''.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update PhishIntentionLLM for the new split/class-folder dataset."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-name", default="phishing_dataset")
    parser.add_argument(
        "--remove-old-raw",
        action="store_true",
        help="Remove data/raw/phish_iris and data/raw/putra after backing up project files.",
    )
    parser.add_argument(
        "--archive-outputs",
        action="store_true",
        help="Archive outputs because predictions reference old sample IDs.",
    )
    parser.add_argument(
        "--create-dataset-symlink",
        action="store_true",
        help="Create data/raw/phishing_dataset pointing to the external dataset root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying files.",
    )
    return parser.parse_args()


def validate_external_layout(dataset_root: Path) -> None:
    missing = []
    for split in SPLITS:
        for class_name in CLASSES:
            folder = dataset_root / "image" / split / class_name
            if not folder.is_dir():
                missing.append(folder)
    if missing:
        raise SystemExit(
            "New dataset does not match the required layout. Missing:\n- "
            + "\n- ".join(str(path) for path in missing)
        )


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists() and not source.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def write_readme_section(readme: Path) -> None:
    start = "<!-- NEW_DATASET_WORKFLOW_START -->"
    end = "<!-- NEW_DATASET_WORKFLOW_END -->"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# PhishIntentionLLM\n"
    if start in existing and end in existing:
        before = existing.split(start, 1)[0].rstrip()
        after = existing.split(end, 1)[1].lstrip()
        updated = before + "\n\n" + README_SECTION + "\n\n" + after
    else:
        updated = existing.rstrip() + "\n\n" + README_SECTION + "\n"
    readme.write_text(updated, encoding="utf-8")


def image_counts(dataset_root: Path) -> dict[str, int]:
    counts = {}
    for split in SPLITS:
        for class_name in CLASSES:
            folder = dataset_root / "image" / split / class_name
            counts[f"{split}/{class_name}"] = sum(
                1 for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    return counts


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    validate_external_layout(dataset_root)

    if not (project_root / "src" / "phishintention").is_dir():
        raise SystemExit(f"Not a PhishIntentionLLM project root: {project_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project_root / "migration_backups" / f"new_dataset_{timestamp}"

    planned = [
        "Back up scripts/prepare_data.py, README.md, manifest and annotation artefacts",
        "Replace scripts/prepare_data.py with the new dataset-only implementation",
        "Archive existing data/annotations because old sample IDs are invalid",
        "Archive existing manifest.csv",
        "Update README.md with the new dataset workflow",
        "Generate a new manifest.csv",
    ]
    if args.remove_old_raw:
        planned.append("Remove data/raw/phish_iris and data/raw/putra")
    if args.archive_outputs:
        planned.append("Archive outputs")
    if args.create_dataset_symlink:
        planned.append("Create data/raw/phishing_dataset symlink")

    print("Planned changes:")
    for item in planned:
        print("-", item)
    print("\nDataset image counts:")
    for key, value in image_counts(dataset_root).items():
        print(f"- {key}: {value}")

    if args.dry_run:
        print("\nDry run only. No files changed.")
        return

    backup_targets = [
        project_root / "scripts" / "prepare_data.py",
        project_root / "README.md",
        project_root / "data" / "processed" / "manifest.csv",
        project_root / "data" / "annotations",
    ]
    if args.archive_outputs:
        backup_targets.append(project_root / "outputs")

    for source in backup_targets:
        if source.exists() or source.is_symlink():
            relative = source.relative_to(project_root)
            copy_if_exists(source, backup_root / relative)

    # Existing annotation artefacts and the manifest refer to old sample IDs.
    annotation_dir = project_root / "data" / "annotations"
    if annotation_dir.exists():
        shutil.rmtree(annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = project_root / "data" / "processed" / "manifest.csv"
    if manifest_path.exists():
        manifest_path.unlink()

    if args.archive_outputs:
        outputs_dir = project_root / "outputs"
        if outputs_dir.exists():
            shutil.rmtree(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

    prepare_path = project_root / "scripts" / "prepare_data.py"
    prepare_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_path.write_text(PREPARE_DATA_SOURCE, encoding="utf-8")

    write_readme_section(project_root / "README.md")

    if args.remove_old_raw:
        for old_folder in (
            project_root / "data" / "raw" / "phish_iris",
            project_root / "data" / "raw" / "putra",
        ):
            if old_folder.exists():
                shutil.rmtree(old_folder)

    if args.create_dataset_symlink:
        link = project_root / "data" / "raw" / "phishing_dataset"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(dataset_root, target_is_directory=True)

    # Build the new manifest directly using the same implementation rules.
    rows = []
    for split in SPLITS:
        for class_name in CLASSES:
            folder = dataset_root / "image" / split / class_name
            for image_path in sorted(folder.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = image_path.relative_to(dataset_root)
                key = f"{args.source_name}:{relative.as_posix()}"
                is_phishing = class_name == "phishing"
                row = {
                    "sample_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                    "source": args.source_name,
                    "split": split,
                    "dataset_class": class_name,
                    "phishing_status": class_name,
                    "image_path": str(image_path.resolve()),
                    "relative_image_path": relative.as_posix(),
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
        raise SystemExit("No supported images were found after validation.")
    if frame["sample_id"].duplicated().any():
        raise SystemExit("Duplicate sample IDs were generated; migration stopped.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)

    report = {
        "project_root": str(project_root),
        "dataset_root": str(dataset_root),
        "backup_root": str(backup_root),
        "manifest": str(manifest_path),
        "total_images": int(len(frame)),
        "counts": {
            f"{split}/{class_name}": int(
                ((frame["split"] == split) & (frame["dataset_class"] == class_name)).sum()
            )
            for split in SPLITS
            for class_name in CLASSES
        },
        "phishing_annotation_candidates": int((frame["phishing_status"] == "phishing").sum()),
        "legitimate_reference_images": int((frame["phishing_status"] == "legitimate").sum()),
    }
    report_path = project_root / "migration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMigration completed.")
    print("Backup:", backup_root)
    print("Manifest:", manifest_path)
    print("Report:", report_path)
    print("Total images:", len(frame))
    print("\nNext commands:")
    print("python -m py_compile scripts/prepare_data.py")
    print("python -m compileall -q src scripts tests")
    print("python -m pytest -q")
    print("python -m streamlit run app.py")


if __name__ == "__main__":
    main()
