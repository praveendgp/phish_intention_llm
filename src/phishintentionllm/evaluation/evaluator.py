"""Score framework predictions against the annotator-generated manifest.

    manifest.jsonl      (annotators)  ->  reference labels
    predictions.jsonl   (framework)   ->  system under test
                          |
                          v
                    metric suite
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..annotation.store import AnnotationStore
from ..config import Config, load_config
from ..schemas import Intention
from .metrics import (agreement_rate, co_occurrence, evaluate, evaluate_binary,
                      sector_matrix)


class Evaluator:
    def __init__(self, config: Optional[Config] = None,
                 store: Optional[AnnotationStore] = None):
        self.cfg = config or load_config()
        self.store = store or AnnotationStore(self.cfg)
        self.thresholds = self.cfg.get("evaluation.complexity_thresholds",
                                       {1: 1, 2: 1, 3: 2})
        self.default_reference = self.cfg.get("evaluation.reference", "manifest")

    # ------------------------------------------------------------------
    def _predictions(self, run_path: Optional[str] = None) -> Dict[str, List[str]]:
        if run_path:
            rows = self.store.load_run(run_path).get("results", [])
        else:
            rows = self.store.load_predictions()
        latest: Dict[str, List[str]] = {}
        for row in rows:
            if row.get("error") or not row.get("sample_id"):
                continue
            latest[row["sample_id"]] = row.get("labels", [])
        return latest

    def _reference(self, reference: Optional[str] = None
                   ) -> tuple[Dict[str, List[str]], str]:
        choice = reference or self.default_reference
        if choice == "ground_truth":
            return self.store.ground_truth_map(), "ground_truth"
        return self.store.manifest_map(), "manifest"

    # ------------------------------------------------------------------
    def evaluate_run(self, run_path: Optional[str] = None,
                     reference: Optional[str] = None,
                     verified_only: bool = False) -> Dict[str, Any]:
        ref_map, ref_name = self._reference(reference)
        pred_map = self._predictions(run_path)

        meta_rows = {r["sample_id"]: r for r in
                     (self.store.load_manifest_rows() if ref_name == "manifest"
                      else self.store.load_ground_truth())
                     if r.get("sample_id")}

        if verified_only and ref_name == "manifest":
            ref_map = {sid: labels for sid, labels in ref_map.items()
                       if meta_rows.get(sid, {}).get("human_verified")}

        shared = [sid for sid in pred_map if sid in ref_map]
        if not shared:
            return {"error": f"No overlap between predictions and the {ref_name}.",
                    "reference": ref_name,
                    "n_reference": len(ref_map), "n_predictions": len(pred_map)}

        y_true = [ref_map[sid] for sid in shared]
        y_pred = [pred_map[sid] for sid in shared]
        report = evaluate(y_true, y_pred, self.thresholds)

        sector_rows = [{"sector": meta_rows.get(sid, {}).get("sector", "other"),
                        "labels": pred_map[sid]} for sid in shared]

        flagged = sum(1 for sid in shared
                      if meta_rows.get(sid, {}).get("needs_review"))
        verified = sum(1 for sid in shared
                       if meta_rows.get(sid, {}).get("human_verified"))

        return {
            "reference": ref_name,
            "n_evaluated": len(shared),
            "n_reference": len(ref_map),
            "n_predictions": len(pred_map),
            "reference_flagged_for_review": flagged,
            "reference_human_verified": verified,
            "report": report.to_dict(),
            "agreement": agreement_rate(y_true, y_pred),
            "credential_theft_benchmark": evaluate_binary(
                y_true, y_pred, Intention.CREDENTIAL_THEFT.value),
            "co_occurrence_predicted": co_occurrence(y_pred),
            "co_occurrence_reference": co_occurrence(y_true),
            "sector_matrix": sector_matrix(sector_rows),
            "sample_ids": shared,
        }

    # ------------------------------------------------------------------
    def per_sample(self, run_path: Optional[str] = None,
                   reference: Optional[str] = None) -> List[Dict[str, Any]]:
        """Row-by-row comparison, useful for error analysis in the UI."""
        ref_map, ref_name = self._reference(reference)
        pred_map = self._predictions(run_path)
        rows: List[Dict[str, Any]] = []
        for sid in sorted(set(pred_map) & set(ref_map)):
            truth, pred = set(ref_map[sid]), set(pred_map[sid])
            if truth == pred:
                outcome = "exact"
            elif truth & pred:
                outcome = "partial"
            else:
                outcome = "miss"
            rows.append({
                "sample_id": sid,
                f"{ref_name} labels": ", ".join(sorted(truth)) or "—",
                "predicted labels": ", ".join(sorted(pred)) or "—",
                "outcome": outcome,
                "missed": ", ".join(sorted(truth - pred)) or "—",
                "extra": ", ".join(sorted(pred - truth)) or "—",
            })
        return rows

    # ------------------------------------------------------------------
    def compare_runs(self, framework_run: str, baseline_run: str,
                     reference: Optional[str] = None) -> Dict[str, Any]:
        """Table 4 analogue: multi-agent framework vs single-agent baseline."""
        return {"PhishIntentionLLM": self.evaluate_run(framework_run, reference),
                "Single-Agent": self.evaluate_run(baseline_run, reference)}

    # ------------------------------------------------------------------
    def manifest_quality(self) -> Dict[str, Any]:
        """Health of the reference itself - annotator agreement and review load."""
        rows = self.store.load_manifest_rows()
        if not rows:
            return {"total": 0}
        agreements: Dict[str, int] = {}
        per_category: Dict[str, int] = {}
        per_count: Dict[str, int] = {}
        for row in rows:
            key = row.get("agreement", "n/a")
            agreements[key] = agreements.get(key, 0) + 1
            for label in row.get("labels", []):
                per_category[label] = per_category.get(label, 0) + 1
            bucket = f"{len(row.get('labels', []))} intention(s)"
            per_count[bucket] = per_count.get(bucket, 0) + 1

        total = len(rows)
        full = agreements.get("full agreement", 0)
        return {
            "total": total,
            "agreement_breakdown": agreements,
            "inter_annotator_agreement": round(full / total, 4) if total else 0.0,
            "tie_breaks": sum(1 for r in rows if r.get("tie_break_used")),
            "needs_review": sum(1 for r in rows if r.get("needs_review")),
            "human_verified": sum(1 for r in rows if r.get("human_verified")),
            "per_category": per_category,
            "per_intention_count": dict(sorted(per_count.items())),
            "co_occurrence": co_occurrence([r.get("labels", []) for r in rows]),
            "sector_matrix": sector_matrix(rows),
        }
