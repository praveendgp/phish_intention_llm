"""PhishIntentionLLM orchestrator - the algorithm of Fig. 3 in the base paper.

    function PHISHINTENTIONLLM(I, K_B, K_C, tau)
        Layer 1  E            <- VISIONANALYSISAGENT(I)
        Layer 2  E_enriched   <- CONTEXTENRICHMENTAGENT(E, RETRIEVEPATTERNS(K_B))
        Layer 3  C, S         <- CLASSIFICATIONAGENT(E_enriched, features)
                 P            <- {c in C | Top-k(S, k=3)}
        Layer 4  A            <- U_{c in P} SPECIALISTAGENT(E_enriched, K_c, c)
        Layer 5  R, conf      <- VALIDATIONAGENT(E, E_enriched, C, S, A)
                 if conf < tau: widen to categories \\ P and re-validate
                 T            <- {(t, evidence, confidence) | confidence >= tau}
                 if |T| = 0  : T <- {argmax_t confidence(t)}
        return T

Every layer receives the screenshot `I` itself: the image is encoded once per
sample and handed to each vision agent, so no layer reasons over a text-only
description of the page.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ..agents.classification_agent import ClassificationAgent
from ..agents.context_agent import ContextEnrichmentAgent
from ..agents.specialist_agents import SpecialistPool
from ..agents.validation_agent import ValidationAgent
from ..agents.vision_agent import VisionAnalysisAgent
from ..config import Config, load_config
from ..llm.ollama_client import OllamaClient
from ..rag.retriever import KnowledgeRetriever
from ..schemas import (AgentStep, AnalysisResult, Intention, IntentionResult,
                       Sample, SpecialistFinding)
from ..utils.image import load_image_b64
from ..utils.logging import get_logger

log = get_logger(__name__)

StepCallback = Optional[Callable[[AgentStep], None]]


class PhishIntentionLLM:
    """Multi-agent RAG framework for phishing intention detection.

    This is the system under evaluation. It never reads the manifest, so its
    predictions remain independent of the reference labels.
    """

    def __init__(self, config: Optional[Config] = None,
                 client: Optional[OllamaClient] = None):
        self.cfg = config or load_config()
        self.client = client or OllamaClient(self.cfg)
        self.retriever = KnowledgeRetriever(self.cfg)
        shared = dict(config=self.cfg, client=self.client, retriever=self.retriever)

        self.vision_agent = VisionAnalysisAgent(**shared)
        self.context_agent = ContextEnrichmentAgent(**shared)
        self.classification_agent = ClassificationAgent(**shared)
        self.specialists = SpecialistPool(**shared)
        self.validation_agent = ValidationAgent(**shared)

        self.tau = float(self.cfg.get("framework.confidence_threshold", 0.6))
        self.max_edge = int(self.cfg.get("framework.max_image_edge", 1280))
        self.feedback_enabled = bool(
            self.cfg.get("framework.enable_feedback_loop", True))

    # ------------------------------------------------------------------
    def analyse_sample(self, sample: Sample,
                       on_step: StepCallback = None) -> AnalysisResult:
        result = self.analyse_screenshot(
            sample.screenshot_path, on_step=on_step, brand_hint=sample.brand,
            sample_id=sample.sample_id, source=sample.source)
        result.is_phishing = result.is_phishing and sample.is_phishing
        return result

    # ------------------------------------------------------------------
    def analyse_screenshot(
        self,
        image_path: str,
        on_step: StepCallback = None,
        brand_hint: Optional[str] = None,
        sample_id: Optional[str] = None,
        source: str = "upload",
    ) -> AnalysisResult:
        started = time.time()
        steps: List[AgentStep] = []
        vision_calls_before = self.client.vision_calls

        def emit(step: AgentStep) -> AgentStep:
            steps.append(step)
            if on_step:
                on_step(step)
            return step

        def touch(step: AgentStep) -> None:
            if on_step:
                on_step(step)

        result = AnalysisResult(sample_id=sample_id or str(image_path),
                                source=source, screenshot_path=str(image_path),
                                steps=steps)

        # The screenshot is encoded ONCE and shared by every vision agent.
        try:
            image_b64 = load_image_b64(image_path, self.max_edge)
        except Exception as exc:
            step = emit(AgentStep("Image Loader", "Layer 0 - Input", "local"))
            step.fail(str(exc))
            result.error = f"Cannot read screenshot: {exc}"
            result.elapsed = round(time.time() - started, 2)
            return result

        try:
            # ---------- Layer 1: Vision Analysis ----------------------
            step = emit(self.vision_agent.new_step())
            elements = self.vision_agent.analyse(image_b64, step, brand_hint)
            touch(step)

            # ---------- Layer 2: Context Enrichment -------------------
            step = emit(self.context_agent.new_step())
            context = self.context_agent.enrich(elements, image_b64, step)
            touch(step)

            # ---------- Layer 3: Classification -----------------------
            step = emit(self.classification_agent.new_step())
            candidates, all_scores, cmeta = self.classification_agent.classify(
                image_b64, context, step)
            touch(step)
            result.category_scores = all_scores

            # ---------- Layer 4: Specialist Analysis ------------------
            findings = self._run_specialists(
                [c.category for c in candidates], context, image_b64, emit, touch)

            # ---------- Layer 5: Validation & Synthesis ---------------
            step = emit(self.validation_agent.new_step())
            intentions, vmeta = self.validation_agent.validate(
                context, candidates, all_scores, findings, image_b64, step)
            touch(step)

            # ---------- Feedback loop (conf < tau) --------------------
            if (self.feedback_enabled
                    and vmeta.get("overall_confidence", 0.0) < self.tau):
                remaining = [c for c in self.specialists.all_categories()
                             if c not in {f.category for f in findings}]
                if remaining:
                    result.feedback_loop_used = True
                    log.info("Confidence %.2f < tau %.2f - widening to %s",
                             vmeta.get("overall_confidence", 0.0), self.tau, remaining)
                    findings += self._run_specialists(
                        remaining, context, image_b64, emit, touch,
                        note="feedback loop")
                    step = emit(self.validation_agent.new_step())
                    step.agent = "Validation & Synthesis Agent (re-run)"
                    intentions, vmeta = self.validation_agent.validate(
                        context, candidates, all_scores, findings, image_b64, step)
                    touch(step)

            # ---------- Result formatting -----------------------------
            result.intentions = self._format_results(intentions, findings)
            result.specialist_findings = findings
            result.sector = vmeta.get("sector") or elements.sector
            result.overall_confidence = vmeta.get("overall_confidence", 0.0)
            result.is_phishing = bool(vmeta.get("is_phishing", True))
            result.phishing_score = vmeta.get("phishing_score", 0.0)
            result.verdict = vmeta.get("verdict", "Phishing website")
            result.risk_summary = vmeta.get("risk_summary", "")
            result.recommended_action = vmeta.get("recommended_action", "")

        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Pipeline failed for %s", image_path)
            result.error = f"{type(exc).__name__}: {exc}"

        result.vision_calls = self.client.vision_calls - vision_calls_before
        result.elapsed = round(time.time() - started, 2)
        return result

    # ------------------------------------------------------------------
    def _run_specialists(self, categories: List[str], context, image_b64: str,
                         emit, touch, note: str = "") -> List[SpecialistFinding]:
        findings: List[SpecialistFinding] = []
        for category in categories:
            agent = self.specialists.get(category)
            step = emit(agent.new_step())
            if note:
                step.layer = f"{step.layer} ({note})"
            try:
                findings.append(agent.analyse(context, image_b64, step))
            except Exception as exc:
                step.fail(str(exc))
            touch(step)

        # Show the complete four-expert panel in the UI by recording skips.
        if not note:
            for category in self.specialists.all_categories():
                if category in categories:
                    continue
                step = emit(self.specialists.get(category).new_step())
                step.skip("Not nominated by the Classification Agent")
                touch(step)
        return findings

    # ------------------------------------------------------------------
    def _format_results(self, intentions: List[IntentionResult],
                        findings: List[SpecialistFinding]) -> List[IntentionResult]:
        """T <- {(t, evidence, conf) | conf >= tau}; ensure |T| >= 1."""
        kept = [i for i in intentions if i.confidence >= self.tau]
        if kept:
            return kept
        if intentions:
            return [max(intentions, key=lambda i: i.confidence)]
        confirmed = [f for f in findings if f.confirmed]
        if confirmed:
            best = max(confirmed, key=lambda f: f.confidence)
            return [IntentionResult(best.category, best.confidence, best.evidence,
                                    best.reasoning[:300])]
        return [IntentionResult(Intention.CREDENTIAL_THEFT.value, 0.0, [],
                                "No intention could be established from this screenshot.")]

    # ------------------------------------------------------------------
    def preflight(self) -> Dict[str, object]:
        """Verify Ollama, that every model is pulled and that all are multimodal."""
        report: Dict[str, object] = {"host": self.client.host, "alive": False,
                                     "models": {}, "missing": [],
                                     "non_vision": []}
        if not self.client.is_alive():
            return report
        report["alive"] = True
        report["available"] = self.client.list_models()

        for spec in self.cfg.all_models():
            ready = self.client.has_model(spec.model)
            capable = self.client.supports_vision(spec.model) if ready else None
            report["models"][spec.role] = {
                "model": spec.model, "label": spec.label, "ready": ready,
                "declared_vision": spec.vision, "detected_vision": capable,
                "stage": ("manifest" if spec.role.startswith(("annotator", "manifest"))
                          else "framework"),
            }
            if not ready and spec.model not in report["missing"]:
                report["missing"].append(spec.model)
            if capable is False and spec.model not in report["non_vision"]:
                report["non_vision"].append(spec.model)
        return report
