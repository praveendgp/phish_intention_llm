"""Persistence for the manifest, framework predictions and manual annotations.

Three distinct artefacts are kept apart on purpose:

  manifest.jsonl           annotator-generated REFERENCE labels
  predictions.jsonl        framework output, scored AGAINST the manifest
  manual_annotations.jsonl human labels from the manual annotation workbench
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config, load_config
from ..schemas import AnalysisResult, ManifestRecord, ManualAnnotation
from ..utils.logging import get_logger

log = get_logger(__name__)
_LOCK = threading.Lock()


def _append(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed line in %s", path.name)
    return rows


class AnnotationStore:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or load_config()
        self.manifest_path = self.cfg.resolve("storage.manifest")
        self.manifest_runs_dir = self.cfg.resolve("storage.manifest_runs_dir")
        self.predictions_path = self.cfg.resolve("storage.predictions")
        self.manual_path = self.cfg.resolve("storage.manual_annotations")
        self.gt_path = self.cfg.resolve("storage.ground_truth")
        self.runs_dir = self.cfg.resolve("storage.runs_dir")

    # ------------------------------------------------------------------
    # MANIFEST (annotator output - the reference labels)
    # ------------------------------------------------------------------
    def append_manifest(self, record: ManifestRecord) -> None:
        _append(self.manifest_path, record.to_dict())

    def load_manifest_rows(self) -> List[Dict[str, Any]]:
        """All manifest lines; later lines supersede earlier ones per sample."""
        latest: Dict[str, Dict[str, Any]] = {}
        for row in _read(self.manifest_path):
            sid = row.get("sample_id")
            if sid:
                latest[sid] = row
        return list(latest.values())

    def load_manifest(self) -> List[ManifestRecord]:
        return [ManifestRecord.from_dict(r) for r in self.load_manifest_rows()]

    def manifest_map(self) -> Dict[str, List[str]]:
        """sample_id -> reference labels. This is what evaluation compares against."""
        return {r["sample_id"]: r.get("labels", [])
                for r in self.load_manifest_rows() if r.get("sample_id")}

    def manifest_ids(self) -> List[str]:
        return [r["sample_id"] for r in self.load_manifest_rows() if r.get("sample_id")]

    def manifest_record(self, sample_id: str) -> Optional[Dict[str, Any]]:
        for row in self.load_manifest_rows():
            if row.get("sample_id") == sample_id:
                return row
        return None

    def rewrite_manifest(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Persist an edited manifest (used after human verification)."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with _LOCK, open(self.manifest_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        log.info("Rewrote manifest with %s record(s)", count)
        return count

    def export_manifest_csv(self, path: str | Path) -> Path:
        """Flat CSV view of the manifest for reports and spreadsheets."""
        import csv

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self.load_manifest_rows()
        fields = ["sample_id", "source", "brand", "split", "screenshot_path",
                  "labels", "n_intentions", "sector", "is_phishing", "agreement",
                  "tie_break_used", "needs_review", "human_verified"]
        with open(target, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    **{k: row.get(k, "") for k in fields},
                    "labels": "|".join(row.get("labels", [])),
                })
        return target

    def save_manifest_run(self, run_id: str, payload: Dict[str, Any]) -> Path:
        self.manifest_runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_runs_dir / f"manifest_run_{run_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def list_manifest_runs(self) -> List[Path]:
        if not self.manifest_runs_dir.exists():
            return []
        return sorted(self.manifest_runs_dir.glob("manifest_run_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    # ------------------------------------------------------------------
    # PREDICTIONS (framework output - the thing being evaluated)
    # ------------------------------------------------------------------
    def save_prediction(self, result: AnalysisResult,
                        run_id: Optional[str] = None) -> None:
        record = result.to_dict()
        record["batch_run_id"] = run_id
        _append(self.predictions_path, record)

    def load_predictions(self) -> List[Dict[str, Any]]:
        return _read(self.predictions_path)

    def prediction_map(self) -> Dict[str, List[str]]:
        latest: Dict[str, List[str]] = {}
        for row in self.load_predictions():
            if row.get("error"):
                continue
            if row.get("sample_id"):
                latest[row["sample_id"]] = row.get("labels", [])
        return latest

    def latest_prediction(self, sample_id: str) -> Optional[Dict[str, Any]]:
        rows = [r for r in self.load_predictions() if r.get("sample_id") == sample_id]
        return rows[-1] if rows else None

    def save_run(self, run_id: str, payload: Dict[str, Any]) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"run_{run_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def list_runs(self) -> List[Path]:
        if not self.runs_dir.exists():
            return []
        return sorted(self.runs_dir.glob("run_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    def load_run(self, path: str | Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # MANUAL annotations (human labels)
    # ------------------------------------------------------------------
    def save_manual(self, annotation: ManualAnnotation) -> None:
        _append(self.manual_path, annotation.to_dict())

    def load_manual(self) -> List[Dict[str, Any]]:
        return _read(self.manual_path)

    def manual_for(self, sample_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.load_manual() if r.get("sample_id") == sample_id]

    # ------------------------------------------------------------------
    # GROUND TRUTH (human-reviewed export)
    # ------------------------------------------------------------------
    def write_ground_truth(self, rows: Iterable[Dict[str, Any]]) -> int:
        self.gt_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with _LOCK, open(self.gt_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        log.info("Wrote %s ground-truth records to %s", count, self.gt_path)
        return count

    def load_ground_truth(self) -> List[Dict[str, Any]]:
        return _read(self.gt_path)

    def ground_truth_map(self) -> Dict[str, List[str]]:
        return {r["sample_id"]: r.get("labels", [])
                for r in self.load_ground_truth() if r.get("sample_id")}

    # ------------------------------------------------------------------
    def reference_map(self, reference: Optional[str] = None) -> Dict[str, List[str]]:
        """The label set the framework is scored against."""
        choice = reference or self.cfg.get("evaluation.reference", "manifest")
        return self.ground_truth_map() if choice == "ground_truth" else self.manifest_map()
