"""Layer 2 - Context Enrichment Agent (semantic layer).

Retrieves basic threat patterns from K_B, tags suspicious elements with security
context, maps visual elements to security implications and produces preliminary
threat hypotheses.

The agent re-examines the screenshot itself rather than trusting the Layer 1
transcript - it is a VLM agent, so it can catch elements the perception layer
under-reported.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AgentStep, EnrichedContext, VisualElements
from ..utils.json_parse import as_float, as_list
from .base import CATEGORY_BLOCK, VisionAgent

SYSTEM = (
    "You are a phishing threat-intelligence analyst. You look at the screenshot "
    "and enrich the observations with security meaning, grounding every "
    "statement in the retrieved knowledge base. You always reply with one valid "
    "JSON object."
)

PROMPT = """## Elements already reported by the perception layer
Sector presented: {sector}
Branding: {branding}
Domain hints: {domains}
Form fields: {fields}
Buttons: {buttons}
Interface elements: {interface}
Layout: {layout}
On-screen text: {text}

## Retrieved basic threat patterns (knowledge base K_B)
{patterns}

## Task
Look at the screenshot yourself, then tag the suspicious elements and map them
to security implications. If you see something the perception layer missed, add
it under "missed_elements".

Return JSON:
{{
  "tagged_elements": [
    {{"element": "<observed element>", "tag": "<deception technique it matches>",
      "implication": "<what an attacker gains from it>"}}
  ],
  "missed_elements": ["<anything visible that was not listed above>"],
  "security_implications": ["<concise implication statements>"],
  "hypotheses": ["<preliminary intention hypotheses, each naming its trigger evidence>"],
  "deception_signals": ["<specific K_B patterns that this page matches>"],
  "phishing_likelihood": <0.0-1.0 float: how likely this page is phishing at all>,
  "notes": "<one sentence on anything ambiguous or unreadable>"
}}

{categories}
Reference only elements you can actually see in the image."""


class ContextEnrichmentAgent(VisionAgent):
    name = "Context Enrichment Agent"
    layer = "Layer 2 - Semantic"
    stage = "framework"
    role = "context"

    def enrich(self, elements: VisualElements, image_b64: str,
               step: Optional[AgentStep] = None) -> EnrichedContext:
        step = step or self.new_step()

        query = " ".join([elements.ocr_text[:600],
                          " ".join(elements.form_fields),
                          " ".join(elements.buttons),
                          " ".join(elements.branding)]).strip() \
            or "generic phishing website screenshot"

        static_patterns = self.retriever.kb.retrieve_patterns(limit_per_facet=4)
        patterns_block = (self.retriever.as_context(query, top_k=6) + "\n"
                          + "\n".join(f"- {p}" for p in static_patterns[:10]))

        prompt = PROMPT.format(
            sector=elements.sector,
            branding=", ".join(elements.branding) or "(none visible)",
            domains=", ".join(elements.domain_hints) or "(none visible)",
            fields=", ".join(elements.form_fields) or "(none)",
            buttons=", ".join(elements.buttons) or "(none)",
            interface=", ".join(elements.interface_elements) or "(none)",
            layout=elements.layout or "(not described)",
            text=elements.ocr_text[:1200] or "(no text read)",
            patterns=patterns_block,
            categories=CATEGORY_BLOCK,
        )

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64, system=SYSTEM,
            default={"security_implications": [], "hypotheses": []},
        )

        # Fold anything the VLM spotted that Layer 1 missed back into the elements.
        missed = as_list(data.get("missed_elements"))[:8]
        if missed:
            elements.interface_elements = list(
                dict.fromkeys(elements.interface_elements + missed))

        context = EnrichedContext(
            elements=elements,
            tagged_elements=self._normalise_tags(data.get("tagged_elements")),
            security_implications=as_list(data.get("security_implications")),
            hypotheses=as_list(data.get("hypotheses")),
            retrieved_patterns=[p.text for p in self.retriever.retrieve(query, 6)],
        )

        step.finish(
            f"Tagged {len(context.tagged_elements)} element(s), raised "
            f"{len(context.hypotheses)} hypothes(e)s from "
            f"{len(context.retrieved_patterns)} retrieved passages"
            + (f"; +{len(missed)} missed element(s)" if missed else ""),
            context=context.to_dict(),
            deception_signals=as_list(data.get("deception_signals")),
            missed_elements=missed,
            phishing_likelihood=as_float(data.get("phishing_likelihood"), 0.5),
            notes=str(data.get("notes", ""))[:400],
        )
        return context

    @staticmethod
    def _normalise_tags(value: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if isinstance(value, dict):
            value = [value]
        for item in value or []:
            if isinstance(item, dict):
                out.append({"element": str(item.get("element", ""))[:200],
                            "tag": str(item.get("tag", ""))[:200],
                            "implication": str(item.get("implication", ""))[:300]})
            elif isinstance(item, str):
                out.append({"element": item[:200], "tag": "", "implication": ""})
        return [t for t in out if t["element"]]
