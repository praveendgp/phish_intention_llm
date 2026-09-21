"""Layer 4 - Specialist Analysis Layer.

Four expert agents, one per threat category. Each is activated only when the
Classification Agent nominates its category (or when the feedback loop widens
the search), and each re-examines the screenshot itself while reasoning inside
its own domain knowledge K_c.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AgentStep, EnrichedContext, Intention, SpecialistFinding
from ..utils.json_parse import as_bool, as_float, as_list
from .base import VisionAgent

SYSTEM_TEMPLATE = (
    "You are the {category} expert agent inside a phishing-intention analysis "
    "framework. You inspect the screenshot and assess ONLY whether this specific "
    "intention is present. You are rigorous: absence of visible evidence means "
    "you reject the hypothesis. Reply with one valid JSON object."
)

PROMPT = """## Your domain: {category}
{definition}

## Specialist knowledge (K_c)
Primary features (F_c):
{features}

Common targets (T_c):
{targets}

Specialised techniques (M_c):
{techniques}

Distinctive indicators (I_c):
{indicators}

Counter-indicators (reject if these dominate):
{counters}

## Enriched page context
Sector: {sector}
Form fields: {fields}
Buttons: {buttons}
Branding: {branding}
Interface elements: {interface}
Security implications: {implications}
On-screen text: {text}

## Task
Look at the screenshot and decide whether "{category}" is truly one of this
page's intentions.

Return JSON:
{{
  "confirmed": true/false,
  "confidence": <0.0-1.0>,
  "evidence": ["<specific visible element supporting your decision>"],
  "matched_indicators": ["<which I_c indicators you can actually see>"],
  "reasoning": "<2-3 sentences of domain-specific analysis>"
}}

Be strict: confirm only when the page's visible design actually serves this goal."""


class SpecialistAgent(VisionAgent):
    """A single category expert."""

    layer = "Layer 4 - Specialist Analysis"
    stage = "framework"
    role = "specialist"

    def __init__(self, category: str, *args, **kwargs):
        self.category = category
        self.name = f"{category} Expert"
        super().__init__(*args, **kwargs)

    def analyse(self, context: EnrichedContext, image_b64: str,
                step: Optional[AgentStep] = None) -> SpecialistFinding:
        step = step or self.new_step()
        knowledge = self.retriever.kb.retrieve_specialist_knowledge(self.category)
        elements = context.elements

        prompt = PROMPT.format(
            category=self.category,
            definition=knowledge.get("definition", ""),
            features=self.bullets(knowledge.get("F_c_primary_features")),
            targets=self.bullets(knowledge.get("T_c_common_targets")),
            techniques=self.bullets(knowledge.get("M_c_techniques")),
            indicators=self.bullets(knowledge.get("I_c_indicators")),
            counters=self.bullets(knowledge.get("counter_indicators")),
            sector=elements.sector,
            fields=", ".join(elements.form_fields) or "(none)",
            buttons=", ".join(elements.buttons) or "(none)",
            branding=", ".join(elements.branding) or "(none)",
            interface=", ".join(elements.interface_elements) or "(none)",
            implications="; ".join(context.security_implications[:6]) or "(none)",
            text=elements.ocr_text[:1000] or "(none)",
        )

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64,
            system=SYSTEM_TEMPLATE.format(category=self.category),
            default={"confirmed": False, "confidence": 0.0},
        )

        confirmed = as_bool(data.get("confirmed"))
        finding = SpecialistFinding(
            category=self.category,
            confirmed=confirmed,
            confidence=as_float(data.get("confidence"), 0.6 if confirmed else 0.2),
            evidence=as_list(data.get("evidence"))[:6],
            reasoning=str(data.get("reasoning", ""))[:800],
            model=self.spec.model,
        )
        step.finish(
            f"{'CONFIRMED' if finding.confirmed else 'rejected'} "
            f"{Intention.short(self.category)} @ {finding.confidence:.2f}",
            finding=finding.to_dict(),
            matched_indicators=as_list(data.get("matched_indicators"))[:6],
        )
        return finding


class SpecialistPool:
    """Lazily instantiated pool of the four expert agents."""

    def __init__(self, config=None, client=None, retriever=None):
        self._shared = dict(config=config, client=client, retriever=retriever)
        self._agents: Dict[str, SpecialistAgent] = {}

    def get(self, category: str) -> SpecialistAgent:
        if category not in self._agents:
            self._agents[category] = SpecialistAgent(category, **self._shared)
        return self._agents[category]

    def all_categories(self) -> List[str]:
        return [i.value for i in Intention.all()]
