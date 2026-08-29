from pathlib import Path
import hashlib
import pandas as pd


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

PUTRA_SCREENSHOT_PRIORITY = [
    "original_js_on.jpg",
    "index_js_on.jpg",
    "clean_js_on.jpg",
    "original_js_off.jpg",
    "index_js_off.jpg",
    "clean_js_off.jpg",
]


def normalise_path(path: Path) -> str:
    """
    Return a normalised lower-case path for reliable path matching.
    """
    return str(path).replace("\\", "/").lower()


def infer_putra_status(path: Path) -> str:
    """
    Infer phishing status from the Zenodo archive/folder name.
    """
    path_text = normalise_path(path)

    legitimate_markers = [
        "not-phishing",
        "not_phishing",
        "notphishing",
    ]

    if any(marker in path_text for marker in legitimate_markers):
        return "legitimate"

    return "phishing"


def find_phish_iris_images(root: str | Path) -> list"""
    Find all supported Phish-IRIS screenshots.

    Phish-IRIS stores one screenshot per brand-class record, so all
    discovered image files are retained.
    """
    root = Path(root)

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def choose_putra_screenshot(screenshots_directory: Path) -> Path | None:
    """
    Select one canonical screenshot for a Putra website record.

    Preference is given to original_js_on.jpg because it is the closest
    representation of the original website with JavaScript enabled.
    Falls back to the other supplied screenshot variants when necessary.
    """
    for filename in PUTRA_SCREENSHOT_PRIORITY:
        candidate = screenshots_directory / filename

        if candidate.exists() and candidate.is_file():
            return candidate

    fallback_images = sorted(
        path
        for path in screenshots_directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    return fallback_images[0] if fallback_images else None


def find_putra_records(root: str | Path) -> list"""
    Discover one record per Putra website folder.

    A valid record is recognised through a directory named 'screenshots'.
    Only one canonical screenshot is selected from each website.
    """
    root = Path(root)

    if not root.exists():
        return []

    records = []

    screenshot_directories = sorted(
        path
        for path in root.rglob("screenshots")
        if path.is_dir()
    )

    for screenshots_directory in screenshot_directories:
        image_path = choose_putra_screenshot(
            screenshots_directory
        )

        if image_path is None:
            continue

        website_directory = screenshots_directory.parent
        website_id = website_directory.name

        records.append(
            {
                "website_id": website_id,
                "image_path": image_path,
                "website_directory": website_directory,
                "asset_details_path": (
                    website_directory / "asset_details.json"
                ),
                "original_html_path": (
                    website_directory / "original.html"
                ),
                "index_html_path": (
                    website_directory / "index.html"
                ),
                "clean_html_path": (
                    website_directory / "clean.html"
                ),
                "phishing_status": infer_putra_status(
                    website_directory
                ),
            }
        )

    return records


def infer_phish_iris_brand(image_path: Path) -> str:
    """
    In Phish-IRIS, the image's parent directory is normally the brand.
    """
    return image_path.parent.name


def infer_phish_iris_status(image_path: Path) -> str:
    """
    Treat the Phish-IRIS 'other' class as legitimate and all brand
    impersonation classes as phishing.
    """
    brand = infer_phish_iris_brand(image_path).lower()

    if brand == "other":
        return "legitimate"

    return "phishing"


def infer_phish_iris_split(image_path: Path) -> str:
    """
    Detect train or validation split from the directory components.
    """
    path_parts = {
        part.lower()
        for part in image_path.parts
    }

    if "val" in path_parts or "validation" in path_parts:
        return "val"

    if "test" in path_parts:
        return "test"

    if "train" in path_parts:
        return "train"

    return "unspecified"


def create_sample_id(
    source: str,
    stable_identifier: str,
) -> str:
    """
    Generate a deterministic sample identifier.
    """
    value = f"{source}:{stable_identifier}"

    return hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:16]


def empty_intention_columns() -> dict:
    """
    Create empty multi-label annotation columns.
    """
    return {
        "credential_theft": "",
        "financial_fraud": "",
        "malware_distribution": "",
        "personal_information_harvesting": "",
        "annotation_status": "unlabelled",
        "annotator": "",
        "notes": "",
    }


def build_phish_iris_rows(
    root: str | Path | None,
) -> list"""
    Build manifest rows for Phish-IRIS.
    """
    if not root:
        return []

    rows = []

    for image_path in find_phish_iris_images(root):
        stable_identifier = str(image_path.resolve())

        row = {
            "sample_id": create_sample_id(
                "phish_iris",
                stable_identifier,
            ),
            "source": "phish_iris",
            "website_id": image_path.stem,
            "image_path": str(image_path.resolve()),
            "screenshot_variant": image_path.name,
            "brand": infer_phish_iris_brand(image_path),
            "phishing_status": infer_phish_iris_status(
                image_path
            ),
            "split": infer_phish_iris_split(image_path),
            "asset_details_path": "",
            "original_html_path": "",
            "index_html_path": "",
            "clean_html_path": "",
        }

        row.update(empty_intention_columns())
        rows.append(row)

    return rows


def build_putra_rows(
    root: str | Path | None,
) -> list"""
    Build one manifest row per Putra website.
    """
    if not root:
        return []

    rows = []

    for record in find_putra_records(root):
        website_id = record["website_id"]
        website_directory = record["website_directory"]
        image_path = record["image_path"]

        stable_identifier = (
            f"{record['phishing_status']}:"
            f"{website_id}:"
            f"{website_directory.resolve()}"
        )

        row = {
            "sample_id": create_sample_id(
                "putra",
                stable_identifier,
            ),
            "source": "putra",
            "website_id": website_id,
            "image_path": str(image_path.resolve()),
            "screenshot_variant": image_path.name,
            "brand": "",
            "phishing_status": record["phishing_status"],
            "split": "unspecified",
            "asset_details_path": (
                str(record["asset_details_path"].resolve())
                if record["asset_details_path"].exists()
                else ""
            ),
            "original_html_path": (
                str(record["original_html_path"].resolve())
                if record["original_html_path"].exists()
                else ""
            ),
            "index_html_path": (
                str(record["index_html_path"].resolve())
                if record["index_html_path"].exists()
                else ""
            ),
            "clean_html_path": (
                str(record["clean_html_path"].resolve())
                if record["clean_html_path"].exists()
                else ""
            ),
        }

        row.update(empty_intention_columns())
        rows.append(row)

    return rows


def build_manifest(
    phish_iris: str | Path | None,
    putra: str | Path | None,
    out: str | Path,
) -> pd.DataFrame:
    """
    Build a combined Phish-IRIS and Putra manifest.
    """
    rows = []

    rows.extend(
        build_phish_iris_rows(phish_iris)
    )

    rows.extend(
        build_putra_rows(putra)
    )

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        dataframe = dataframe.drop_duplicates(
            subset=["sample_id"],
            keep="first",
        )

        dataframe = dataframe.sort_values(
            by=["source", "phishing_status", "sample_id"]
        ).reset_index(drop=True)

    output_path = Path(out)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe