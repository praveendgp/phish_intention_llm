"""Manifest builder - orchestrates the annotator ensemble.

The manifest is the **reference label file**. It is produced once per sample by
the annotator ensemble and is later used to measure the framework's performance.

    load screenshot
        ├─ Annotator A (VLM) ──┐
        ├─ Annotator B (VLM) ──┤  analyse_agreement()
        ├─ Tie-Breaker  (VLM) ─┤  only when a category is genuinely disputed
        └─ Finaliser    (VLM) ─┘  writes one ManifestRecord
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..config import Config, load_config
from ..llm.ollama_client import OllamaClient
from ..rag.retriever import KnowledgeRetriever
from ..schemas import AgentStep, AnnotatorVote, Intention, ManifestRecord, Sample
from ..utils.image import load_image_b64
from ..utils.logging import get_logger
from .annotators import (AnnotatorAgent, FinalizerAgent, TieBreakerAgent,
                         analyse_agreement)
from .store import AnnotationStore

log = get_logger(__name__)

StepCallback = Optional[Callable[[AgentStep], None]]
ProgressCallback = Optional[Callable[[int, int, ManifestRecord], None]]


class ManifestBuilder:
    """Runs the annotator ensemble and writes `manifest.jsonl`."""

    def __init__(self, config: Optional[Config] = None,
                 client: Optional[OllamaClient] = None,
                 store: Optional[AnnotationStore] = None):
        self.cfg = config or load_config()
        self.client = client or OllamaClient(self.cfg)
        self.retriever = KnowledgeRetriever(self.cfg)
        self.store = store or AnnotationStore(self.cfg)

        shared = dict(config=self.cfg, client=self.client, retriever=self.retriever)
        self.annotator_a = AnnotatorAgent(model_role="annotator_a", **shared)
        self.annotator_b = AnnotatorAgent(model_role="annotator_b", **shared)
        self.tiebreaker = TieBreakerAgent(**shared)
        self.finalizer = FinalizerAgent(**shared)

        settings = self.cfg.get("manifest", {}) or {}
        self.margin = float(settings.get("agreement_margin", 0.15))
        self.enable_tiebreaker = bool(settings.get("enable_tiebreaker", True))
        self.enable_finalizer = bool(settings.get("enable_finalizer", True))
        self.min_confidence = float(settings.get("min_confidence", 0.35))
        self.max_intentions = int(settings.get("max_intentions", 3))
        self.review_triggers = list(settings.get("flag_for_review_when", []))
        self.max_edge = int(self.cfg.get("framework.max_image_edge", 1280))

    # ------------------------------------------------------------------
    def annotate_sample(self, sample: Sample, on_step: StepCallback = None,
                        run_id: str = "") -> ManifestRecord:
        started = time.time()
        steps: List[AgentStep] = []

        def emit(step: AgentStep) -> AgentStep:
            steps.append(step)
            if on_step:
                on_step(step)
            return step

        def touch(step: AgentStep) -> None:
            if on_step:
                on_step(step)

        record = ManifestRecord(
            sample_id=sample.sample_id, source=sample.source,
            screenshot_path=sample.screenshot_path, brand=sample.brand,
            split=sample.split, is_phishing=sample.is_phishing,
            manifest_run_id=run_id, steps=steps,
        )

        try:
            image_b64 = load_image_b64(sample.screenshot_path, self.max_edge)
        except Exception as exc:
            step = emit(AgentStep("Image Loader", "Manifest - Input", "local",
                                  stage="manifest"))
            step.fail(str(exc))
            record.error = f"Cannot read screenshot: {exc}"
            record.needs_review = True
            record.review_reasons = ["unreadable screenshot"]
            record.elapsed = round(time.time() - started, 2)
            return record

        # ---- 1. two independent annotations --------------------------
        votes: List[AnnotatorVote] = []
        for agent in (self.annotator_a, self.annotator_b):
            step = emit(agent.new_step())
            try:
                votes.append(agent.annotate(image_b64, step))
            except Exception as exc:
                step.fail(str(exc))
                votes.append(AnnotatorVote(annotator=agent.name,
                                           model=agent.spec.model,
                                           error=str(exc)))
            touch(step)

        record.votes = votes
        record.annotators = [v.annotator for v in votes]

        usable = [v for v in votes if v.error is None]
        if not usable:
            record.error = "Both annotators failed."
            record.needs_review = True
            record.review_reasons = ["annotation failed"]
            record.elapsed = round(time.time() - started, 2)
            return record

        # ---- 2. agreement analysis -----------------------------------
        agreement = analyse_agreement(votes, self.margin)
        record.agreement = agreement["status"]
        record.agreed = agreement["agreed"]
        record.disputed = agreement["disputed"]

        # ---- 3. tie-break only when genuinely required ----------------
        tiebreak: Optional[Dict[str, Any]] = None
        step = emit(self.tiebreaker.new_step())
        if (self.enable_tiebreaker and agreement["requires_tiebreak"]
                and agreement["tiebreak_categories"]):
            try:
                tiebreak = self.tiebreaker.resolve(
                    votes, agreement["tiebreak_categories"], image_b64, step)
                record.tie_break_used = True
                record.tie_break_decisions = tiebreak["decisions"]
            except Exception as exc:
                step.fail(str(exc))
        else:
            step.skip(f"Not required - {agreement['status']}")
        touch(step)

        # ---- 4. finalisation -----------------------------------------
        step = emit(self.finalizer.new_step())
        if self.enable_finalizer:
            try:
                payload = self.finalizer.finalize(votes, agreement, image_b64,
                                                  tiebreak, step)
            except Exception as exc:
                step.fail(str(exc))
                payload = self._fallback_payload(votes, agreement, tiebreak)
        else:
            step.skip("Finaliser disabled in config")
            payload = self._fallback_payload(votes, agreement, tiebreak)
        touch(step)

        # ---- 5. assemble the manifest record --------------------------
        labels = [c for c in payload["labels"]
                  if payload["confidences"].get(c, 0.0) >= self.min_confidence]
        if not labels and payload["labels"]:
            labels = payload["labels"][:1]        # never emit an empty phishing row

        record.labels = labels[: self.max_intentions]
        record.confidences = {c: round(payload["confidences"].get(c, 0.0), 4)
                              for c in record.labels}
        record.evidence = {c: payload["evidence"].get(c, []) for c in record.labels}
        record.sector = payload["sector"]
        record.is_phishing = bool(payload["is_phishing"] and sample.is_phishing)
        record.finalizer_summary = payload["summary"]

        reasons: List[str] = []
        if payload.get("needs_human_review"):
            reasons.append("finaliser flagged it")
        if "conflict" in self.review_triggers and agreement["status"] == "conflict":
            reasons.append("annotators conflicted")
        if "tie-break applied" in self.review_triggers and record.tie_break_used:
            reasons.append("tie-break applied")
        if ("low confidence" in self.review_triggers and record.confidences
                and max(record.confidences.values()) < 0.6):
            reasons.append("low confidence")
        if payload.get("label_quality") == "low":
            reasons.append("low label quality")
        record.needs_review = bool(reasons)
        record.review_reasons = reasons

        record.elapsed = round(time.time() - started, 2)
        return record

    # ------------------------------------------------------------------
    def _fallback_payload(self, votes: List[AnnotatorVote],
                          agreement: Dict[str, Any],
                          tiebreak: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic consensus used when the finaliser is unavailable."""
        labels = list(agreement.get("agreed", []))
        confidences: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}

        for category in labels:
            confidences[category] = round(sum(
                v.confidences.get(category, 0.0) for v in votes
                if v.error is None) / max(1, len([v for v in votes if not v.error])), 4)
            evidence[category] = sum((v.evidence.get(category, []) for v in votes), [])[:6]

        if tiebreak:
            for category, keep in (tiebreak.get("decisions") or {}).items():
                if keep and category not in labels:
                    labels.append(category)
                    confidences[category] = tiebreak.get("confidences", {}).get(
                        category, 0.6)
                    evidence[category] = (tiebreak.get("evidence", {}) or {}).get(
                        category, [])

        if not labels:      # last resort: the most confident single claim
            best: Optional[str] = None
            best_conf = 0.0
            for vote in votes:
                for category, conf in vote.confidences.items():
                    if conf > best_conf:
                        best, best_conf = category, conf
            if best:
                labels = [best]
                confidences[best] = best_conf

        usable = [v for v in votes if v.error is None]
        return {
            "labels": sorted(labels, key=lambda c: confidences.get(c, 0), reverse=True),
            "confidences": confidences,
            "evidence": evidence,
            "rejected": [],
            "sector": usable[0].sector if usable else "other",
            "is_phishing": any(v.is_phishing for v in usable) if usable else True,
            "label_quality": "medium",
            "needs_human_review": True,
            "summary": "Deterministic consensus (finaliser unavailable).",
        }

    # ------------------------------------------------------------------
    def build(self, samples: List[Sample], on_progress: ProgressCallback = None,
              on_step: StepCallback = None,
              stop_flag: Optional[Callable[[], bool]] = None,
              save: bool = True, overwrite: bool = False) -> Dict[str, Any]:
        """Annotate many samples and append them to the manifest file."""
        run_id = uuid.uuid4().hex[:12]
        started = time.time()
        records: List[ManifestRecord] = []

        existing = set() if overwrite else set(self.store.manifest_ids())

        for index, sample in enumerate(samples, 1):
            if stop_flag and stop_flag():
                log.info("Manifest run %s stopped after %s sample(s)", run_id, index - 1)
                break
            if sample.sample_id in existing:
                continue
            try:
                record = self.annotate_sample(sample, on_step=on_step, run_id=run_id)
            except Exception as exc:              # keep the batch alive
                log.error("Annotation failed for %s: %s", sample.sample_id, exc)
                record = ManifestRecord(sample_id=sample.sample_id,
                                        source=sample.source,
                                        screenshot_path=sample.screenshot_path,
                                        manifest_run_id=run_id, error=str(exc),
                                        needs_review=True)
            records.append(record)
            if save:
                self.store.append_manifest(record)
            if on_progress:
                on_progress(index, len(samples), record)

        payload = self._summarise(run_id, records, time.time() - started)
        if save:
            self.store.save_manifest_run(run_id, payload)
        return payload

    # ------------------------------------------------------------------
    def _summarise(self, run_id: str, records: List[ManifestRecord],
                   elapsed: float) -> Dict[str, Any]:
        ok = [r for r in records if not r.error]
        per_category: Dict[str, int] = {}
        per_count: Dict[str, int] = {}
        agreements: Dict[str, int] = {}

        for record in ok:
            for label in record.labels:
                per_category[label] = per_category.get(label, 0) + 1
            key = f"{record.n_intentions} intention(s)"
            per_count[key] = per_count.get(key, 0) + 1
            agreements[record.agreement] = agreements.get(record.agreement, 0) + 1

        return {
            "manifest_run_id": run_id,
            "created_at": time.time(),
            "elapsed": round(elapsed, 2),
            "n_annotated": len(records),
            "n_success": len(ok),
            "n_failed": len(records) - len(ok),
            "avg_seconds_per_sample": round(elapsed / max(1, len(records)), 2),
            "per_category": per_category,
            "per_intention_count": dict(sorted(per_count.items())),
            "agreement_breakdown": agreements,
            "tie_breaks": sum(1 for r in ok if r.tie_break_used),
            "needs_review": sum(1 for r in ok if r.needs_review),
            "annotator_models": {
                "annotator_a": self.annotator_a.spec.model,
                "annotator_b": self.annotator_b.spec.model,
                "tiebreaker": self.tiebreaker.spec.model,
                "finalizer": self.finalizer.spec.model,
            },
            "records": [r.to_dict() for r in records],
        }

    # ------------------------------------------------------------------
    def preflight(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {"host": self.client.host, "alive": False,
                                  "models": {}, "missing": []}
        if not self.client.is_alive():
            return report
        report["alive"] = True
        for agent in (self.annotator_a, self.annotator_b,
                      self.tiebreaker, self.finalizer):
            ready = self.client.has_model(agent.spec.model)
            report["models"][agent.name] = {"model": agent.spec.model, "ready": ready}
            if not ready and agent.spec.model not in report["missing"]:
                report["missing"].append(agent.spec.model)
        return report
