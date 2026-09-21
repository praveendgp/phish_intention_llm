"""Layer 3 - Classification Agent.

Implements CLASSIFICATIONAGENT(E_enriched, features) from the paper's algorithm:
multi-label classification producing the candidate set C with confidence scores
S, from which Top-k(S, k<=3) categories are selected for specialist analysis.

This agent is part of the FRAMEWORK stage. It is deliberately independent of the
annotator ensemble that produces the manifest - the framework must never see the
reference labels it will later be scored against.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import AgentStep, CategoryScore, EnrichedContext, Intention
from ..utils.json_parse import as_float, as_list
from .base import CATEGORY_BLOCK, VisionAgent

SYSTEM = (
    "You are the classification layer of a phishing-intention framework. You "
    "examine the screenshot and score each of the four intention categories, "
    "then nominate the most likely ones for deeper expert analysis. Reply with "
    "one valid JSON object."
)

PROMPT = """## Enriched analysis from the earlier layers
Sector: {sector}
Branding: {branding}
Domain hints: {domains}
Form fields: {fields}
Buttons: {buttons}
Interface elements: {interface}
Security implications: {implications}
Preliminary hypotheses: {hypotheses}
On-screen text: {text}

## Category features retrieved from the knowledge base (F_c)
{features}

{categories}

## Task
Look at the screenshot and score ALL FOUR categories, then nominate the top
candidates (at most {top_k}) that deserve specialist analysis.

Return JSON:
{{
  "scores": {{
    "Credential Theft": <0.0-1.0>,
    "Financial Fraud": <0.0-1.0>,
    "Malware Distribution": <0.0-1.0>,
    "Personal Information Harvesting": <0.0-1.0>
  }},
  "evidence": {{
    "<category name>": ["<visible element supporting that score>"]
  }},
  "nominated": ["<categories to send to the specialists, highest first>"],
  "sector": "<industry the page imitates>",
  "reasoning": "<2-3 sentences>"
}}

Score every category, including the ones you believe are absent (give them a
low score). Nominate between 1 and {top_k} categories."""


class ClassificationAgent(VisionAgent):
    name = "Classification Agent"
    layer = "Layer 3 - Classification"
    stage = "framework"
    role = "classifier"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = int(self.cfg.get("framework.top_k_categories", 3))
        self.min_categories = int(self.cfg.get("framework.min_categories", 1))

    # ------------------------------------------------------------------
    def classify(self, image_b64: str, context: EnrichedContext,
                 step: Optional[AgentStep] = None
                 ) -> Tuple[List[CategoryScore], List[CategoryScore], Dict[str, Any]]:
        """Return (candidates P, all scores S, metadata)."""
        step = step or self.new_step()
        elements = context.elements

        features = self.retriever.kb.retrieve_category_features()
        features_block = "\n".join(
            f"{name}:\n" + "\n".join(f"  - {f}" for f in items[:5])
            for name, items in features.items()
        )

        prompt = PROMPT.format(
            sector=elements.sector,
            branding=", ".join(elements.branding) or "(none)",
            domains=", ".join(elements.domain_hints) or "(none)",
            fields=", ".join(elements.form_fields) or "(none)",
            buttons=", ".join(elements.buttons) or "(none)",
            interface=", ".join(elements.interface_elements) or "(none)",
            implications="; ".join(context.security_implications[:6]) or "(none)",
            hypotheses="; ".join(context.hypotheses[:5]) or "(none)",
            text=elements.ocr_text[:1100] or "(none)",
            features=features_block,
            categories=CATEGORY_BLOCK,
            top_k=self.top_k,
        )

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64, system=SYSTEM,
            default={"scores": {}},
        )

        all_scores = self._parse_scores(data)
        candidates = self._select_candidates(all_scores, data)

        step.finish(
            "Nominated " + (", ".join(f"{Intention.short(c.category)} {c.score:.2f}"
                                      for c in candidates) or "nothing"),
            scores=[s.to_dict() for s in all_scores],
            candidates=[c.category for c in candidates],
            sector=data.get("sector", elements.sector),
            reasoning=str(data.get("reasoning", ""))[:600],
        )
        meta = {"sector": data.get("sector", elements.sector),
                "reasoning": str(data.get("reasoning", ""))[:600]}
        return candidates, all_scores, meta

    # ------------------------------------------------------------------
    def _parse_scores(self, data: Dict[str, Any]) -> List[CategoryScore]:
        raw = data.get("scores") or data.get("categories") or {}
        evidence_map = data.get("evidence") or {}

        scores: Dict[str, CategoryScore] = {}

        def put(name: Any, value: Any, evidence: Any = None) -> None:
            category = Intention.coerce(name)
            if category is None:
                return
            confidence = as_float(value, 0.0)
            ev = as_list(evidence)[:6]
            current = scores.get(category.value)
            if current is None or confidence > current.score:
                scores[category.value] = CategoryScore(category.value, confidence, ev)

        if isinstance(raw, dict):
            for name, value in raw.items():
                ev = evidence_map.get(name) if isinstance(evidence_map, dict) else None
                put(name, value, ev)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    put(item.get("category") or item.get("name"),
                        item.get("score") or item.get("confidence"),
                        item.get("evidence"))
                elif isinstance(item, str):
                    put(item, 0.7)

        # Ensure all four categories are represented in S.
        for intention in Intention.all():
            scores.setdefault(intention.value, CategoryScore(intention.value, 0.0, []))

        return sorted(scores.values(), key=lambda s: s.score, reverse=True)

    def _select_candidates(self, all_scores: List[CategoryScore],
                           data: Dict[str, Any]) -> List[CategoryScore]:
        lookup = {s.category: s for s in all_scores}
        nominated: List[CategoryScore] = []
        for name in as_list(data.get("nominated")):
            category = Intention.coerce(name)
            if category and category.value in lookup:
                score = lookup[category.value]
                if score not in nominated:
                    nominated.append(score)

        if not nominated:   # fall back to Top-k(S, k)
            nominated = [s for s in all_scores if s.score > 0][: self.top_k]
        if not nominated:
            nominated = all_scores[: max(1, self.min_categories)]
        return nominated[: self.top_k]
