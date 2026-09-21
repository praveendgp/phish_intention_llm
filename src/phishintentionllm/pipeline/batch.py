"""Batch prediction runner for the framework (and the single-agent baseline)."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from ..annotation.store import AnnotationStore
from ..config import Config, load_config
from ..evaluation.metrics import co_occurrence, evaluate, sector_matrix
from ..schemas import AnalysisResult, Sample
from ..utils.logging import get_logger
from .orchestrator import PhishIntentionLLM
from .single_agent import SingleAgentBaseline

log = get_logger(__name__)

ProgressCallback = Optional[Callable[[int, int, AnalysisResult], None]]


class BatchRunner:
    def __init__(self, config: Optional[Config] = None,
                 framework: Optional[PhishIntentionLLM] = None,
                 store: Optional[AnnotationStore] = None):
        self.cfg = config or load_config()
        self.framework = framework or PhishIntentionLLM(self.cfg)
        self.store = store or AnnotationStore(self.cfg)

    # ------------------------------------------------------------------
    def run(self, samples: List[Sample], mode: str = "framework",
            on_progress: ProgressCallback = None,
            on_step: Optional[Callable] = None,
            stop_flag: Optional[Callable[[], bool]] = None,
            save: bool = True) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        engine = (self.framework if mode == "framework"
                  else SingleAgentBaseline(self.cfg, self.framework.client))
        started = time.time()
        results: List[AnalysisResult] = []

        for index, sample in enumerate(samples, 1):
            if stop_flag and stop_flag():
                log.info("Batch %s stopped after %s sample(s)", run_id, index - 1)
                break
            try:
                result = engine.analyse_sample(sample, on_step=on_step)
            except Exception as exc:
                log.error("Sample %s failed: %s", sample.sample_id, exc)
                result = AnalysisResult(sample_id=sample.sample_id,
                                        source=sample.source,
                                        screenshot_path=sample.screenshot_path,
                                        error=str(exc))
            results.append(result)
            if save:
                self.store.save_prediction(result, run_id)
            if on_progress:
                on_progress(index, len(samples), result)

        payload = self._summarise(run_id, mode, results, time.time() - started)
        if save:
            log.info("Batch %s written to %s", run_id,
                     self.store.save_run(run_id, payload))
        return payload

    # ------------------------------------------------------------------
    def _summarise(self, run_id: str, mode: str, results: List[AnalysisResult],
                   elapsed: float) -> Dict[str, Any]:
        ok = [r for r in results if not r.error]
        rows = [{"sample_id": r.sample_id, "source": r.source,
                 "sector": r.sector, "labels": r.labels} for r in ok]

        # Immediate scoring against the manifest, when it covers these samples.
        reference = self.store.reference_map()
        matched = [(reference[r.sample_id], r.labels) for r in ok
                   if r.sample_id in reference]
        evaluation = None
        if matched:
            evaluation = evaluate([m[0] for m in matched], [m[1] for m in matched],
                                  self.cfg.get("evaluation.complexity_thresholds")
                                  ).to_dict()

        per_category: Dict[str, int] = {}
        for result in ok:
            for label in result.labels:
                per_category[label] = per_category.get(label, 0) + 1

        return {
            "run_id": run_id, "mode": mode, "created_at": time.time(),
            "elapsed": round(elapsed, 2),
            "n_requested": len(results), "n_success": len(ok),
            "n_failed": len(results) - len(ok),
            "avg_seconds_per_sample": round(elapsed / max(1, len(results)), 2),
            "total_vision_calls": sum(r.vision_calls for r in ok),
            "per_category": per_category,
            "co_occurrence": co_occurrence([r.labels for r in ok]),
            "sector_matrix": sector_matrix(rows),
            "feedback_loops": sum(1 for r in ok if r.feedback_loop_used),
            "n_scored_against_manifest": len(matched),
            "evaluation": evaluation,
            "results": [r.to_dict() for r in results],
        }
