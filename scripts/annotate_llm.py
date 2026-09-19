from __future__ import annotations

import argparse
import json
import sys
import time
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


# Explicitly load the .env file from the project root.
load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


from phishintention.annotators import (
    GeminiAnnotator,
    OpenAIAnnotator,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create independent screenshot annotations using "
            "OpenAI or Gemini."
        )
    )

    parser.add_argument(
        "--provider",
        choices=[
            "openai",
            "gemini",
        ],
        required=True,
        help="Annotation provider to use.",
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Path to the project manifest.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/annotations",
        help="Directory in which annotations are stored.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of manifest rows to process.",
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model override. If omitted, the model configured "
            "in .env is used."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace the provider annotation file and process the "
            "selected rows again."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay in seconds between API requests.",
    )

    return parser.parse_args()


def load_completed_sample_ids(
    annotation_path: Path,
) -> set[str]:
    """
    Return sample IDs already present in a provider annotation file.

    This allows an interrupted run to resume without repeating completed
    API requests.
    """
    if not annotation_path.exists():
        return set()

    completed = set()

    for line in annotation_path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        sample_id = item.get("sample_id")

        if sample_id:
            completed.add(str(sample_id))

    return completed


def create_annotator(
    provider: str,
    model: str | None,
):
    """
    Create the selected annotation client.
    """
    if provider == "openai":
        return OpenAIAnnotator(
            model=model,
        )

    return GeminiAnnotator(
        model=model,
    )


def record_failure(
    failure_path: Path,
    sample_id: str,
    image_path: str,
    source: str,
    provider: str,
    model: str,
    error: Exception,
) -> None:
    """
    Append one annotation failure to a JSONL file.
    """
    failure_record = {
        "sample_id": sample_id,
        "image_path": image_path,
        "source": source,
        "provider": provider,
        "model": model,
        "error_type": type(error).__name__,
        "error": str(error),
    }

    with failure_path.open(
        "a",
        encoding="utf-8",
    ) as failure_handle:
        failure_handle.write(
            json.dumps(
                failure_record,
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> None:
    arguments = parse_arguments()

    manifest_path = Path(
        arguments.manifest
    )

    if not manifest_path.is_absolute():
        manifest_path = (
            PROJECT_ROOT / manifest_path
        )

    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found: {manifest_path}"
        )

    dataframe = pd.read_csv(
        manifest_path
    ).fillna("")

    required_columns = {
        "sample_id",
        "source",
        "image_path",
        "phishing_status",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise SystemExit(
            "The manifest is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # The intention annotation experiment operates only on known
    # phishing records.
    phishing_status = (
        dataframe["phishing_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        phishing_status == "phishing"
    ].copy()

    if arguments.limit is not None:
        dataframe = dataframe.head(
            arguments.limit
        )

    if dataframe.empty:
        raise SystemExit(
            "No phishing rows were found in the manifest."
        )

    output_directory = Path(
        arguments.output_dir
    )

    if not output_directory.is_absolute():
        output_directory = (
            PROJECT_ROOT
            / output_directory
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_path = (
        output_directory
        / (
            f"{arguments.provider}"
            "_annotations.jsonl"
        )
    )

    failure_path = (
        output_directory
        / (
            f"{arguments.provider}"
            "_failures.jsonl"
        )
    )

    completed_sample_ids = (
        set()
        if arguments.force
        else load_completed_sample_ids(
            annotation_path
        )
    )

    annotator = create_annotator(
        provider=arguments.provider,
        model=arguments.model,
    )

    file_mode = (
        "w"
        if arguments.force
        else "a"
    )

    total_rows = len(dataframe)
    processed_count = 0
    skipped_count = 0
    failure_count = 0

    with annotation_path.open(
        file_mode,
        encoding="utf-8",
    ) as annotation_handle:
        for position, row in enumerate(
            dataframe.itertuples(
                index=False
            ),
            start=1,
        ):
            sample_id = str(
                row.sample_id
            )

            source = str(
                row.source
            )

            image_path_object = Path(
                str(row.image_path)
            ).expanduser()

            if not image_path_object.is_absolute():
                image_path_object = (
                    PROJECT_ROOT
                    / image_path_object
                )

            image_path_object = (
                image_path_object.resolve()
            )

            image_path = str(
                image_path_object
            )

            if sample_id in completed_sample_ids:
                skipped_count += 1

                print(
                    f"\n[{position}/{total_rows}] "
                    "SKIPPED"
                    f"\nImage: {image_path}"
                    f"\nSample ID: {sample_id}"
                    "\nReason: Annotation already exists."
                )

                continue

            if not image_path_object.exists():
                error = FileNotFoundError(
                    f"Screenshot does not exist: "
                    f"{image_path}"
                )

                record_failure(
                    failure_path=failure_path,
                    sample_id=sample_id,
                    image_path=image_path,
                    source=source,
                    provider=arguments.provider,
                    model=annotator.model,
                    error=error,
                )

                failure_count += 1

                print(
                    f"\n[{position}/{total_rows}] "
                    "FAILED"
                    f"\nImage: {image_path}"
                    f"\nSample ID: {sample_id}"
                    f"\nError: {error}",
                    file=sys.stderr,
                )

                continue

            try:
                annotation, metadata = (
                    annotator.annotate(
                        image_path
                    )
                )

                annotation_record = {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "source": source,
                    "annotation": (
                        annotation.model_dump(
                            mode="json"
                        )
                    ),
                    "metadata": metadata,
                }

                annotation_handle.write(
                    json.dumps(
                        annotation_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                annotation_handle.flush()

                completed_sample_ids.add(
                    sample_id
                )

                processed_count += 1

                print(
                    f"\n[{position}/{total_rows}] "
                    "ANNOTATED"
                    f"\nImage: {image_path}"
                    f"\nSample ID: {sample_id}"
                    f"\nProvider: "
                    f"{arguments.provider}"
                    f"\nModel: {annotator.model}"
                    f"\nLabels: "
                    f"{annotation.binary_labels()}"
                    f"\nImage quality: "
                    f"{annotation.image_quality}"
                )

            except Exception as error:
                record_failure(
                    failure_path=failure_path,
                    sample_id=sample_id,
                    image_path=image_path,
                    source=source,
                    provider=arguments.provider,
                    model=annotator.model,
                    error=error,
                )

                failure_count += 1

                print(
                    f"\n[{position}/{total_rows}] "
                    "FAILED"
                    f"\nImage: {image_path}"
                    f"\nSample ID: {sample_id}"
                    f"\nProvider: "
                    f"{arguments.provider}"
                    f"\nModel: {annotator.model}"
                    f"\nError type: "
                    f"{type(error).__name__}"
                    f"\nError: {error}",
                    file=sys.stderr,
                )

            if arguments.sleep > 0:
                time.sleep(
                    arguments.sleep
                )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ANNOTATION RUN COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Provider: {arguments.provider}"
    )

    print(
        f"Model: {annotator.model}"
    )

    print(
        f"Selected manifest rows: "
        f"{total_rows}"
    )

    print(
        f"New annotations: "
        f"{processed_count}"
    )

    print(
        f"Existing annotations skipped: "
        f"{skipped_count}"
    )

    print(
        f"Failures: {failure_count}"
    )

    print(
        f"Annotation file: "
        f"{annotation_path}"
    )

    if failure_count:
        print(
            f"Failure file: "
            f"{failure_path}"
        )


if __name__ == "__main__":
    main()