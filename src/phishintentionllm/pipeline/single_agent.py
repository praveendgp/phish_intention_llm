"""Single-agent baseline (Section 6.2 of the base paper).

One monolithic VLM prompt, one model, no RAG, no specialists, no multi-layer
reasoning. Used to quantify the gain contributed by the multi-agent RAG
architecture. It is scored against the same manifest as the full framework.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..config import Config, load_config
from ..llm.ollama_client import OllamaClient
from ..schemas import AgentStep, AnalysisResult, Intention, IntentionResult, Sample
from ..utils.image import load_image_b64
from ..utils.json_parse import as_bool, as_float, as_list, parse_json
from ..utils.logging import get_logger

log = get_logger(__name__)

SYSTEM = ("You are a cybersecurity analyst identifying the malicious intentions "
          "of phishing websites from screenshots. Reply with valid JSON only.")

PROMPT = """Look at this website screenshot and identify its phishing intention(s).

Categories:
1. Credential Theft
2. Financial Fraud
3. Malware Distribution
4. Personal Information Harvesting

Return JSON:
{"intentions": [{"category": "<name>", "confidence": <0.0-1.0>,
                 "evidence": ["<what you see>"]}],
 "sector": "<industry>", "is_phishing": true/false,
 "phishing_score": <0.0-1.0>, "verdict": "<one-line summary>"}"""


class SingleAgentBaseline:
    def __init__(self, config: Optional[Config] = None,
                 client: Optional[OllamaClient] = None,
                 model_role: str = "validator"):
        self.cfg = config or load_config()
        self.client = client or OllamaClient(self.cfg)
        self.spec = self.cfg.model(model_role)
        self.max_edge = int(self.cfg.get("framework.max_image_edge", 1280))

    def analyse_screenshot(self, image_path: str, on_step=None,
                           sample_id: Optional[str] = None,
                           source: str = "upload") -> AnalysisResult:
        started = time.time()
        step = AgentStep(agent="Single-Agent Baseline", layer="Baseline",
                         model=self.spec.label or self.spec.model, vision=True)
        steps = [step]
        if on_step:
            on_step(step)

        result = AnalysisResult(sample_id=sample_id or str(image_path),
                                source=source, screenshot_path=str(image_path),
                                steps=steps)
        step.start()
        try:
            image_b64 = load_image_b64(image_path, self.max_edge)
            response = self.client.generate(self.spec, PROMPT, system=SYSTEM,
                                            images=[image_b64], json_mode=True)
            data: Dict[str, Any] = parse_json(response.text, {"intentions": []})

            intentions: List[IntentionResult] = []
            for item in data.get("intentions", []) or []:
                if isinstance(item, str):
                    category, confidence, evidence = Intention.coerce(item), 0.7, []
                elif isinstance(item, dict):
                    category = Intention.coerce(item.get("category"))
                    confidence = as_float(item.get("confidence"), 0.6)
                    evidence = as_list(item.get("evidence"))
                else:
                    continue
                if category:
                    intentions.append(IntentionResult(category.value, confidence,
                                                      evidence[:5]))
            if not intentions:
                intentions = [IntentionResult(Intention.CREDENTIAL_THEFT.value,
                                              0.3, [], "Baseline default")]

            result.intentions = intentions[:3]
            result.sector = str(data.get("sector", "other"))
            result.is_phishing = as_bool(data.get("is_phishing"), True)
            result.phishing_score = as_float(data.get("phishing_score"), 0.7)
            result.verdict = str(data.get("verdict", ""))[:200] or "Phishing website"
            result.overall_confidence = round(
                sum(i.confidence for i in intentions) / max(1, len(intentions)), 4)
            result.vision_calls = 1
            step.finish("Baseline labels: "
                        + ", ".join(i.short for i in result.intentions))
        except Exception as exc:
            log.error("Single-agent baseline failed: %s", exc)
            step.fail(str(exc))
            result.error = str(exc)

        result.elapsed = round(time.time() - started, 2)
        return result

    def analyse_sample(self, sample: Sample, on_step=None) -> AnalysisResult:
        return self.analyse_screenshot(sample.screenshot_path, on_step,
                                       sample.sample_id, sample.source)
