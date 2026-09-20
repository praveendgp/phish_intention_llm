from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


from phishintention.annotators import OllamaAnnotator


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create independent screenshot annotations "
            "using a local Ollama vision model."
        )
    )

    parser.add_argument(
        "--provider-name",
        required=True,
        help=(
            "Short provider identifier, for example "
            "gemma or minicpm."
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Ollama vision model name."
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/processed/manifest.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/annotations"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(
    value: str,
) -> Path:
    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_completed_ids(
    path: Path,
) -> set[str]:
    if not path.exists():
        return set()

    completed = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(
                line
            )

        except json.JSONDecodeError:
            continue

        sample_id = item.get(
            "sample_id"
        )

        if sample_id:
            completed.add(
                str(sample_id)
            )

    return completed


def append_jsonl(
    path: Path,
    record: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        handle.flush()


def main() -> None:
    arguments = parse_arguments()

    manifest_path = resolve_path(
        arguments.manifest
    )

    output_directory = resolve_path(
        arguments.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    dataframe = pd.read_csv(
        manifest_path
    ).fillna("")

    phishing_mask = (
        dataframe[
            "phishing_status"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("phishing")
    )

    dataframe = dataframe[
        phishing_mask
    ].copy()

    if arguments.limit is not None:
        dataframe = dataframe.head(
            arguments.limit
        )

    annotation_path = (
        output_directory
        / (
            f"ollama_"
            f"{arguments.provider_name}"
            "_annotations.jsonl"
        )
    )

    failure_path = (
        output_directory
        / (
            f"ollama_"
            f"{arguments.provider_name}"
            "_failures.jsonl"
        )
    )

    if arguments.force:
        annotation_path.write_text(
            "",
            encoding="utf-8",
        )

        failure_path.write_text(
            "",
            encoding="utf-8",
        )

        completed_ids = set()

    else:
        completed_ids = (
            load_completed_ids(
                annotation_path
            )
        )

    annotator = OllamaAnnotator(
        model=arguments.model,
        provider_name=(
            arguments.provider_name
        ),
    )

    total = len(dataframe)
    success_count = 0
    skipped_count = 0
    failure_count = 0

    for position, row in enumerate(
        dataframe.itertuples(
            index=False
        ),
        start=1,
    ):
        sample_id = str(
            row.sample_id
        )

        image_path = Path(
            str(
                row.image_path
            )
        ).expanduser()

        if not image_path.is_absolute():
            image_path = (
                PROJECT_ROOT
                / image_path
            )

        image_path = (
            image_path.resolve()
        )

        if sample_id in completed_ids:
            skipped_count += 1

            print(
                f"\n[{position}/{total}] "
                "SKIPPED"
                f"\nImage: {image_path}"
                f"\nSample ID: {sample_id}",
                flush=True,
            )

            continue

        print(
            f"\n[{position}/{total}] "
            "ANNOTATING"
            f"\nProvider: "
            f"{arguments.provider_name}"
            f"\nModel: "
            f"{arguments.model}"
            f"\nImage: {image_path}"
            f"\nSample ID: {sample_id}",
            flush=True,
        )

        try:
            annotation, metadata = (
                annotator.annotate(
                    image_path
                )
            )

            record = {
                "sample_id": sample_id,
                "image_path": str(
                    image_path
                ),
                "source": str(
                    row.source
                ),
                "annotation": (
                    annotation.model_dump(
                        mode="json"
                    )
                ),
                "metadata": metadata,
            }

            append_jsonl(
                annotation_path,
                record,
            )

            completed_ids.add(
                sample_id
            )

            success_count += 1

            print(
                "Annotation saved.",
                flush=True,
            )

        except Exception as error:
            append_jsonl(
                failure_path,
                {
                    "sample_id": sample_id,
                    "image_path": str(
                        image_path
                    ),
                    "source": str(
                        row.source
                    ),
                    "provider": (
                        arguments.provider_name
                    ),
                    "model": arguments.model,
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                },
            )

            failure_count += 1

            print(
                "Annotation failed."
                f"\nError: {error}",
                file=sys.stderr,
                flush=True,
            )

    print("\n" + "=" * 72)
    print("LOCAL ANNOTATION COMPLETE")
    print("=" * 72)
    print(
        f"Provider: "
        f"{arguments.provider_name}"
    )
    print(
        f"Model: {arguments.model}"
    )
    print(
        f"New annotations: "
        f"{success_count}"
    )
    print(
        f"Skipped: {skipped_count}"
    )
    print(
        f"Failures: {failure_count}"
    )
    print(
        f"Output: {annotation_path}"
    )


if __name__ == "__main__":
    main()