#!/usr/bin/env python3
"""Safely archive obsolete cloud-model annotation code.

This script is conservative by design:
- defaults to dry-run;
- creates a timestamped backup before changing files;
- archives obsolete modules/scripts instead of permanently deleting them;
- removes cloud SDK requirements and cloud-only environment variables;
- updates annotators/__init__.py only when matching import blocks or __all__ entries exist;
- scans active code after cleanup and reports remaining cloud references;
- optionally runs compileall and pytest.

Run from the PhishIntentionLLM project root.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

CLOUD_TERMS = ("openai", "gemini", "groq")
CLOUD_ENV_PREFIXES = ("OPENAI_", "GEMINI_", "GROQ_")
CLOUD_REQUIREMENTS = {
    "openai",
    "google-genai",
    "google-generativeai",
    "groq",
}

# Known obsolete source files from earlier cloud-assisted workflows.
DEFAULT_ARCHIVE_PATHS = (
    "src/phishintention/annotators/openai_annotator.py",
    "src/phishintention/annotators/gemini_annotator.py",
    "src/phishintention/annotators/groq_adjudicator.py",
    "scripts/annotate_llm.py",
    "scripts/groq_adjudicate.py",
    "scripts/retry_failed_groq.py",
)

# Old generated results are historical artefacts. Archive them, do not silently delete.
DEFAULT_ARCHIVE_GLOBS = (
    "data/annotations/openai_*",
    "data/annotations/gemini_*",
    "data/annotations/groq_*",
    "data/annotations/archive/groq/**",
)

ACTIVE_SCAN_ROOTS = (
    "src",
    "scripts",
    "tests",
    "app.py",
    "requirements.txt",
    "pyproject.toml",
    ".env.example",
    "README.md",
)

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".csv"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive obsolete OpenAI/Gemini/Groq code and clean active configuration."
    )
    parser.add_argument("--project-root", default=".")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this option, the script performs a dry run.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request a dry run. This is the default.",
    )
    parser.add_argument(
        "--archive-generated",
        action="store_true",
        help="Also archive generated OpenAI/Gemini/Groq annotation artefacts.",
    )
    parser.add_argument(
        "--clean-env",
        action="store_true",
        help="Remove OPENAI_*, GEMINI_* and GROQ_* lines from .env and .env.example.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run compileall and pytest after an applied cleanup.",
    )
    parser.add_argument(
        "--permanent-delete",
        action="store_true",
        help="Permanently delete matched obsolete files after backing them up instead of archiving them.",
    )
    return parser.parse_args()


def is_project_root(root: Path) -> bool:
    return (root / "src" / "phishintention").is_dir() and (root / "scripts").is_dir()


def copy_to_backup(root: Path, source: Path, backup_root: Path) -> None:
    relative = source.relative_to(root)
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination, follow_symlinks=False)


def archive_or_delete(
    root: Path,
    source: Path,
    archive_root: Path,
    permanent_delete: bool,
) -> str:
    relative = source.relative_to(root)
    if permanent_delete:
        if source.is_dir() and not source.is_symlink():
            shutil.rmtree(source)
        else:
            source.unlink()
        return f"deleted {relative}"

    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(str(source), str(destination))
    return f"archived {relative} -> {destination.relative_to(root)}"


def dependency_name(line: str) -> str:
    value = line.strip()
    if not value or value.startswith("#"):
        return ""
    value = re.split(r"[<>=!~;\[]", value, maxsplit=1)[0]
    return value.strip().lower().replace("_", "-")


def clean_requirements_text(text: str) -> tuple[str, list[str]]:
    kept = []
    removed = []
    for line in text.splitlines():
        if dependency_name(line) in CLOUD_REQUIREMENTS:
            removed.append(line)
        else:
            kept.append(line)
    return "\n".join(kept).rstrip() + "\n", removed


def clean_env_text(text: str) -> tuple[str, list[str]]:
    kept = []
    removed = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(CLOUD_ENV_PREFIXES):
            removed.append(line)
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n", removed


def clean_annotator_init_text(text: str) -> tuple[str, list[str]]:
    original = text
    removed = []

    # Remove single-line imports and parenthesised import blocks for obsolete modules.
    for module_name in ("openai_annotator", "gemini_annotator", "groq_adjudicator"):
        patterns = [
            rf"(?m)^from\s+\.{re.escape(module_name)}\s+import\s+[^\n]+\n?",
            rf"(?ms)^from\s+\.{re.escape(module_name)}\s+import\s*\(.*?\)\s*\n?",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                removed.append(match.group(0).strip())
                text = re.sub(pattern, "", text)

    obsolete_exports = (
        "OpenAIAnnotator",
        "GeminiAnnotator",
        "GroqAdjudicator",
        "GroqAdjudicationOutput",
    )
    for export in obsolete_exports:
        pattern = rf"(?m)^\s*[\"']{re.escape(export)}[\"']\s*,?\s*\n?"
        match = re.search(pattern, text)
        if match:
            removed.append(match.group(0).strip())
            text = re.sub(pattern, "", text)

    # Collapse excessive blank lines introduced by import removal.
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    if text == original:
        return original, []
    return text, removed


def remove_cloud_sections_from_readme(text: str) -> tuple[str, list[str]]:
    """Remove only explicitly marked legacy cloud sections.

    The script will not heuristically delete normal prose from README because that could
    remove methodology/history unintentionally. Mark obsolete blocks with:
      <!-- LEGACY_CLOUD_MODELS_START -->
      ...
      <!-- LEGACY_CLOUD_MODELS_END -->
    """
    pattern = re.compile(
        r"(?ms)\n?<!-- LEGACY_CLOUD_MODELS_START -->.*?<!-- LEGACY_CLOUD_MODELS_END -->\n?"
    )
    matches = pattern.findall(text)
    updated = pattern.sub("\n", text)
    updated = re.sub(r"\n{3,}", "\n\n", updated).rstrip() + "\n"
    return updated, ["removed marked legacy cloud section" for _ in matches]


def iter_scan_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for item in ACTIVE_SCAN_ROOTS:
        path = root / item
        if not path.exists():
            continue
        if path.is_file():
            if path not in seen:
                seen.add(path)
                yield path
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            if any(part in {"__pycache__", ".venv", "archive", "migration_backups"} for part in candidate.parts):
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def scan_cloud_references(root: Path) -> list[dict[str, object]]:
    findings = []
    regex = re.compile(r"openai|gemini|groq", re.IGNORECASE)
    for path in iter_scan_files(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, start=1):
            if regex.search(line):
                findings.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": number,
                        "text": line.strip()[:500],
                    }
                )
    return findings


def run_command(command: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not is_project_root(root):
        raise SystemExit(f"Not a PhishIntentionLLM project root: {root}")

    apply_changes = bool(args.apply)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / "migration_backups" / f"cloud_cleanup_{timestamp}"
    archive_root = root / "archive" / "cloud_models" / timestamp

    candidates: list[Path] = []
    for relative in DEFAULT_ARCHIVE_PATHS:
        path = root / relative
        if path.exists() or path.is_symlink():
            candidates.append(path)

    if args.archive_generated:
        for pattern in DEFAULT_ARCHIVE_GLOBS:
            for path in root.glob(pattern):
                if path.exists() or path.is_symlink():
                    candidates.append(path)

    # Preserve ordering while removing duplicate paths.
    candidates = list(dict.fromkeys(candidates))

    edits: dict[Path, tuple[str, list[str]]] = {}

    requirements = root / "requirements.txt"
    if requirements.exists():
        cleaned, removed = clean_requirements_text(requirements.read_text(encoding="utf-8"))
        if removed:
            edits[requirements] = (cleaned, removed)

    annotator_init = root / "src" / "phishintention" / "annotators" / "__init__.py"
    if annotator_init.exists():
        cleaned, removed = clean_annotator_init_text(annotator_init.read_text(encoding="utf-8"))
        if removed:
            edits[annotator_init] = (cleaned, removed)

    if args.clean_env:
        for env_name in (".env", ".env.example"):
            env_path = root / env_name
            if env_path.exists():
                cleaned, removed = clean_env_text(env_path.read_text(encoding="utf-8"))
                if removed:
                    edits[env_path] = (cleaned, removed)

    readme = root / "README.md"
    if readme.exists():
        cleaned, removed = remove_cloud_sections_from_readme(readme.read_text(encoding="utf-8"))
        if removed:
            edits[readme] = (cleaned, removed)

    findings_before = scan_cloud_references(root)

    print("Mode:", "APPLY" if apply_changes else "DRY RUN")
    print("\nFiles to archive or delete:")
    if candidates:
        for path in candidates:
            print("-", path.relative_to(root))
    else:
        print("- none")

    print("\nFiles to edit:")
    if edits:
        for path, (_, removed) in edits.items():
            print(f"- {path.relative_to(root)}")
            for item in removed:
                print("    remove:", item)
    else:
        print("- none")

    print(f"\nActive cloud-reference findings before cleanup: {len(findings_before)}")
    for finding in findings_before[:30]:
        print(f"- {finding['file']}:{finding['line']}: {finding['text']}")
    if len(findings_before) > 30:
        print(f"- ... {len(findings_before) - 30} more")

    if not apply_changes:
        print("\nDry run complete. Re-run with --apply after reviewing the plan.")
        return

    # Create backups for everything that is about to change.
    for path in candidates:
        copy_to_backup(root, path, backup_root)
    for path in edits:
        copy_to_backup(root, path, backup_root)

    actions = []
    for path in candidates:
        actions.append(
            archive_or_delete(root, path, archive_root, args.permanent_delete)
        )

    for path, (cleaned, _) in edits.items():
        path.write_text(cleaned, encoding="utf-8")
        actions.append(f"edited {path.relative_to(root)}")

    findings_after = scan_cloud_references(root)

    report = {
        "timestamp": timestamp,
        "project_root": str(root),
        "backup_root": str(backup_root),
        "archive_root": str(archive_root),
        "permanent_delete": bool(args.permanent_delete),
        "archive_generated": bool(args.archive_generated),
        "clean_env": bool(args.clean_env),
        "actions": actions,
        "remaining_cloud_references": findings_after,
    }

    report_path = root / "cloud_cleanup_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    test_results = []
    if args.run_tests:
        commands = [
            [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
            [sys.executable, "-m", "pytest", "-q"],
        ]
        for command in commands:
            code, output = run_command(command, root)
            test_results.append((command, code, output))
            print("\n$", " ".join(command))
            print(output.rstrip())
            if code != 0:
                print(f"Command failed with exit code {code}.", file=sys.stderr)

    print("\nCleanup applied.")
    print("Backup:", backup_root)
    if not args.permanent_delete:
        print("Archive:", archive_root)
    print("Report:", report_path)
    print("Remaining active cloud-reference findings:", len(findings_after))
    for finding in findings_after[:50]:
        print(f"- {finding['file']}:{finding['line']}: {finding['text']}")

    if findings_after:
        print(
            "\nReview remaining references manually. Some may be historical documentation "
            "or compatibility text rather than active code."
        )

    if args.run_tests and any(code != 0 for _, code, _ in test_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
