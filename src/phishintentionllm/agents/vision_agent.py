"""Layer 1 - Vision Analysis Agent (perception layer).

Extracts raw text, interface components, page layout and domain information
directly from the screenshot using a vision-language model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import AgentStep, VisualElements
from ..utils.json_parse import as_bool, as_list
from .base import VisionAgent

SYSTEM = (
    "You are a meticulous web-page perception engine for a cybersecurity "
    "laboratory. You describe exactly what is visible in a screenshot. "
    "You never speculate about intent, never invent text that is not shown, "
    "and you always reply with a single valid JSON object."
)

PROMPT = """Analyse this website screenshot and report ONLY what is visually present.

Return JSON with exactly these keys:
{{
  "ocr_text": "<all readable on-screen text, joined with ' | ' (max 1200 chars)>",
  "interface_elements": ["<UI components: navbar, modal, login card, banner, table, ...>"],
  "form_fields": ["<every input field label or placeholder, e.g. 'Password', 'Card number', 'Date of birth'>"],
  "buttons": ["<exact text on every button or primary link>"],
  "branding": ["<brand names / logos / trademarks visible>"],
  "layout": "<2-3 sentence description of the page structure and visual quality>",
  "domain_hints": ["<any URL, domain, address-bar text or email domain visible>"],
  "has_password_field": true/false,
  "has_payment_field": true/false,
  "has_download_action": true/false,
  "has_identity_fields": true/false,
  "sector": "<one of: {sectors}>",
  "image_quality": "<clear | partial | blank | error-page>"
}}

Rules:
- Transcribe text verbatim; do not translate or summarise it.
- If a field type is absent, use false and an empty list - do not guess.
- "sector" is the industry the page presents itself as belonging to.
{brand_hint}"""


class VisionAnalysisAgent(VisionAgent):
    name = "Vision Analysis Agent"
    layer = "Layer 1 - Perception"
    stage = "framework"
    role = "vision"

    def analyse(self, image_b64: str, step: Optional[AgentStep] = None,
                brand_hint: Optional[str] = None) -> VisualElements:
        step = step or self.new_step()
        hint = (f"- Dataset metadata suggests the impersonated brand may be "
                f"'{brand_hint}'. Confirm it only if you can actually see it."
                if brand_hint else "")
        prompt = PROMPT.format(sectors=", ".join(self.cfg.sectors), brand_hint=hint)

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64, system=SYSTEM,
            default={"ocr_text": "", "layout": ""},
        )

        elements = VisualElements(
            ocr_text=str(data.get("ocr_text", ""))[:2000],
            interface_elements=as_list(data.get("interface_elements")),
            form_fields=as_list(data.get("form_fields")),
            buttons=as_list(data.get("buttons")),
            branding=as_list(data.get("branding")),
            layout=str(data.get("layout", ""))[:800],
            domain_hints=as_list(data.get("domain_hints")),
            sector=self._normalise_sector(data.get("sector")),
            raw={
                "has_password_field": as_bool(data.get("has_password_field")),
                "has_payment_field": as_bool(data.get("has_payment_field")),
                "has_download_action": as_bool(data.get("has_download_action")),
                "has_identity_fields": as_bool(data.get("has_identity_fields")),
                "image_quality": str(data.get("image_quality", "clear")),
            },
        )

        flags = [k.replace("has_", "").replace("_", " ")
                 for k, v in elements.raw.items() if v is True]
        step.finish(
            f"Saw {len(elements.ocr_text)} chars of text, "
            f"{len(elements.form_fields)} form field(s), "
            f"{len(elements.buttons)} button(s); sector='{elements.sector}'"
            + (f"; flags: {', '.join(flags)}" if flags else ""),
            elements=elements.to_dict(),
            latency=data.get("_meta", {}).get("latency"),
        )
        return elements

    def _normalise_sector(self, value: Any) -> str:
        sectors: List[str] = self.cfg.sectors
        text = str(value or "other").strip().lower()
        for sector in sectors:
            if sector.lower() == text:
                return sector
        for sector in sectors:
            if sector.lower() in text or text in sector.lower():
                return sector
        aliases = {
            "bank": "financial", "banking": "financial", "finance": "financial",
            "shopping": "e-commerce", "retail": "e-commerce",
            "social": "social networking", "webmail": "email provider",
            "mail": "email provider", "cloud": "online/cloud service",
            "technology": "online/cloud service", "shipping": "delivery & couriers",
            "logistics": "delivery & couriers", "courier": "delivery & couriers",
            "telecom": "telecommunications", "crypto": "cryptocurrency",
            "payment": "payment services", "govt": "government",
        }
        for key, sector in aliases.items():
            if key in text:
                return sector
        return "other"
