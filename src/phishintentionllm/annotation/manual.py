"""Manual annotation module (retained from the original project design).

Replicates the ground-truth construction protocol of the base paper:

    * two engineers label each sample independently,
    * a third engineer reviews every labelled sample for consistency,
    * a sample may carry one or several intentions.

It also doubles as the **human verification path for the manifest**: a reviewer
can confirm or correct an annotator-generated manifest row, after which the row
is marked `human_verified`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ..config import Config, load_config
from ..schemas import Intention, ManualAnnotation, Sample
from ..utils.logging import get_logger
from .store import AnnotationStore

log = get_logger(__name__)


class ManualAnnotationManager:
    def __init__(self, config: Optional[Config] = None,
                 store: Optional[AnnotationStore] = None):
        self.cfg = config or load_config()
        self.store = store or AnnotationStore(self.cfg)
        settings = self.cfg.get("manual_annotation", {}) or {}
        self.enabled = bool(settings.get("enabled", True))
        self.annotators: List[str] = list(
            settings.get("annotators", ["engineer_1", "engineer_2", "reviewer"]))
        self.require_reviewer = bool(settings.get("require_reviewer", True))
        self.max_intentions = int(settings.get("max_intentions_per_sample", 4))

    # ------------------------------------------------------------------
    @property
    def reviewer(self) -> str:
        return self.annotators[-1] if self.annotators else "reviewer"

    @property
    def labelers(self) -> List[str]:
        return ([a for a in self.annotators if a != self.reviewer]
                if self.require_reviewer else list(self.annotators))

    # ------------------------------------------------------------------
    def annotate(self, sample_id: str, annotator: str, categories: List[str],
                 sector: str = "other", notes: str = "",
                 is_phishing: bool = True) -> ManualAnnotation:
        if not self.enabled:
            raise RuntimeError("Manual annotation is disabled in config.yaml")

        clean: List[str] = []
        for raw in categories:
            category = Intention.coerce(raw)
            if category and category.value not in clean:
                clean.append(category.value)

        annotation = ManualAnnotation(
            sample_id=sample_id, annotator=annotator,
            categories=clean[: self.max_intentions], sector=sector,
            notes=notes.strip()[:2000], is_phishing=is_phishing,
            reviewed=(annotator == self.reviewer),
            reviewer=self.reviewer if annotator == self.reviewer else None,
        )
        self.store.save_manual(annotation)
        log.info("Manual annotation: %s by %s -> %s", sample_id, annotator, clean)
        return annotation

    # ------------------------------------------------------------------
    def latest_by_annotator(self, sample_id: str) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for row in self.store.manual_for(sample_id):
            who = row.get("annotator", "?")
            current = latest.get(who)
            if current is None or row.get("created_at", 0) >= current.get("created_at", 0):
                latest[who] = row
        return latest

    def status(self, sample_id: str) -> Dict[str, Any]:
        latest = self.latest_by_annotator(sample_id)
        done = [a for a in self.labelers if a in latest]
        reviewed = self.reviewer in latest
        if reviewed:
            state = "reviewed"
        elif len(done) >= len(self.labelers):
            state = "awaiting review"
        elif done:
            state = "partially labelled"
        else:
            state = "unlabelled"

        agreement: Optional[bool] = None
        if len(done) >= 2:
            sets = [set(latest[a].get("categories", [])) for a in done[:2]]
            agreement = sets[0] == sets[1]

        return {"sample_id": sample_id, "state": state, "labelled_by": done,
                "reviewed": reviewed, "annotator_agreement": agreement,
                "records": latest}

    # ------------------------------------------------------------------
    def consensus(self, sample_id: str) -> Tuple[List[str], str]:
        latest = self.latest_by_annotator(sample_id)
        if not latest:
            return [], "none"
        if self.reviewer in latest:
            return list(latest[self.reviewer].get("categories", [])), "reviewer"

        label_sets = [set(r.get("categories", [])) for r in latest.values()]
        if len(label_sets) == 1:
            return sorted(label_sets[0]), "single annotator"
        if all(s == label_sets[0] for s in label_sets):
            return sorted(label_sets[0]), "unanimous"

        counts = Counter(c for s in label_sets for c in s)
        majority = sorted(c for c, n in counts.items() if n >= 2)
        if majority:
            return majority, "majority"
        return sorted(set().union(*label_sets)), "union (unresolved)"

    # ------------------------------------------------------------------
    def build_ground_truth(self, samples: Optional[List[Sample]] = None,
                           require_review: Optional[bool] = None) -> List[Dict[str, Any]]:
        require = self.require_reviewer if require_review is None else require_review
        index = {s.sample_id: s for s in (samples or [])}

        by_sample: Dict[str, bool] = {}
        for row in self.store.load_manual():
            by_sample[row.get("sample_id", "")] = True

        rows: List[Dict[str, Any]] = []
        for sample_id in by_sample:
            if not sample_id:
                continue
            status = self.status(sample_id)
            if require and not status["reviewed"]:
                continue
            labels, basis = self.consensus(sample_id)
            if not labels:
                continue
            sample = index.get(sample_id)
            latest = status["records"]
            sector = next((r.get("sector") for r in latest.values()
                           if r.get("sector")), "other")
            rows.append({
                "sample_id": sample_id, "labels": labels,
                "n_intentions": len(labels), "sector": sector, "basis": basis,
                "annotators": sorted(latest), "reviewed": status["reviewed"],
                "source": sample.source if sample else sample_id.split("::")[0],
                "screenshot_path": sample.screenshot_path if sample else "",
            })
        return rows

    def export_ground_truth(self, samples: Optional[List[Sample]] = None) -> int:
        return self.store.write_ground_truth(self.build_ground_truth(samples))

    # ------------------------------------------------------------------
    # Human verification of the annotator-generated manifest
    # ------------------------------------------------------------------
    def verify_manifest_row(self, sample_id: str, labels: List[str],
                            reviewer: Optional[str] = None,
                            sector: Optional[str] = None,
                            notes: str = "") -> bool:
        """Correct/confirm a manifest row and mark it human-verified."""
        rows = self.store.load_manifest_rows()
        clean = []
        for raw in labels:
            category = Intention.coerce(raw)
            if category and category.value not in clean:
                clean.append(category.value)

        found = False
        for row in rows:
            if row.get("sample_id") != sample_id:
                continue
            found = True
            row["labels"] = clean
            row["n_intentions"] = len(clean)
            row["human_verified"] = True
            row["needs_review"] = False
            row["verified_by"] = reviewer or self.reviewer
            row["verification_notes"] = notes[:1000]
            if sector:
                row["sector"] = sector
            row["confidences"] = {c: max(row.get("confidences", {}).get(c, 0.0), 0.99)
                                  for c in clean}
        if found:
            self.store.rewrite_manifest(rows)
            self.annotate(sample_id, reviewer or self.reviewer, clean,
                          sector or "other", notes or "manifest verification")
        return found

    # ------------------------------------------------------------------
    def distribution(self, source: str = "ground_truth") -> Dict[str, Any]:
        """Table 1 analogue: intention counts and intentions-per-record."""
        if source == "manifest":
            rows = self.store.load_manifest_rows()
        else:
            rows = self.store.load_ground_truth() or self.build_ground_truth()
        per_category: Counter = Counter()
        per_count: Counter = Counter()
        for row in rows:
            labels = row.get("labels", [])
            per_category.update(labels)
            per_count[len(labels)] += 1
        return {"total": len(rows), "per_category": dict(per_category),
                "per_intention_count": {f"{k} intention(s)": v
                                        for k, v in sorted(per_count.items())}}
