"""Layer 5 - Validation and Synthesis Agent.

Combines specialist findings, resolves conflicts between competing hypotheses,
weighs evidence by reliability and produces the final classification with
complete evidence chains and confidence scores - while looking at the
screenshot one last time to sanity-check the conclusion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..schemas import (AgentStep, CategoryScore, EnrichedContext, Intention,
                       IntentionResult, SpecialistFinding)
from ..utils.json_parse import as_bool, as_float, as_list
from .base import CATEGORY_BLOCK, VisionAgent

SYSTEM = (
    "You are the validation and synthesis authority of a phishing-intention "
    "framework. You look at the screenshot, weigh every prior analysis, resolve "
    "contradictions and issue the final, defensible verdict. You write for a "
    "human security analyst. Reply with one valid JSON object."
)

PROMPT = """## Candidate classification (Layer 3)
{candidates}

## All category scores
{scores}

## Specialist findings (Layer 4)
{findings}

## Page evidence
Sector: {sector}
Form fields: {fields}
Buttons: {buttons}
Branding: {branding}
Domain hints: {domains}
Security implications: {implications}
On-screen text: {text}

{categories}

## Task
Look at the screenshot and issue the final assessment.

Return JSON:
{{
  "final_intentions": [
    {{"category": "<exact category name>",
      "confidence": <0.0-1.0>,
      "evidence": ["<full evidence chain, most decisive first>"],
      "explanation": "<one plain-English sentence a non-expert can understand>"}}
  ],
  "is_phishing": true/false,
  "phishing_score": <0.0-1.0>,
  "verdict": "<one short headline, e.g. 'Fake Microsoft 365 login harvesting credentials'>",
  "sector": "<final sector>",
  "overall_confidence": <0.0-1.0>,
  "risk_summary": "<2-4 sentences: what the attacker wants, who is at risk, what happens if a victim complies>",
  "recommended_action": "<one concrete defensive action>",
  "conflicts_resolved": ["<any contradiction you had to settle>"]
}}

Rules:
- Include a category only if a specialist confirmed it or the image evidence is decisive.
- Weigh a confirming specialist above a raw classification score.
- Return at least one intention and at most three."""


class ValidationAgent(VisionAgent):
    name = "Validation & Synthesis Agent"
    layer = "Layer 5 - Validation"
    stage = "framework"
    role = "validator"

    def validate(
        self,
        context: EnrichedContext,
        candidates: List[CategoryScore],
        all_scores: List[CategoryScore],
        findings: List[SpecialistFinding],
        image_b64: str,
        step: Optional[AgentStep] = None,
    ) -> Tuple[List[IntentionResult], Dict[str, Any]]:
        step = step or self.new_step()
        elements = context.elements

        prompt = PROMPT.format(
            candidates=self._fmt_candidates(candidates),
            scores="; ".join(f"{Intention.short(s.category)}={s.score:.2f}"
                             for s in all_scores) or "(none)",
            findings=self._fmt_findings(findings),
            sector=elements.sector,
            fields=", ".join(elements.form_fields) or "(none)",
            buttons=", ".join(elements.buttons) or "(none)",
            branding=", ".join(elements.branding) or "(none)",
            domains=", ".join(elements.domain_hints) or "(none)",
            implications="; ".join(context.security_implications[:6]) or "(none)",
            text=elements.ocr_text[:900] or "(none)",
            categories=CATEGORY_BLOCK,
        )

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64, system=SYSTEM,
            default={"final_intentions": []},
        )

        results = self._parse(data, candidates, findings)
        confidence = as_float(data.get("overall_confidence"), 0.0)
        if not confidence and results:
            confidence = round(sum(r.confidence for r in results) / len(results), 4)

        meta = {
            "is_phishing": as_bool(data.get("is_phishing"), True),
            "phishing_score": as_float(data.get("phishing_score"), 0.85),
            "verdict": str(data.get("verdict", ""))[:200] or "Phishing website",
            "sector": str(data.get("sector", elements.sector) or elements.sector),
            "overall_confidence": confidence,
            "risk_summary": str(data.get("risk_summary", ""))[:1200],
            "recommended_action": str(data.get("recommended_action", ""))[:400],
            "conflicts_resolved": as_list(data.get("conflicts_resolved"))[:6],
        }
        step.finish(
            "Final: " + (", ".join(f"{r.short} {r.confidence:.2f}" for r in results)
                         or "none") + f" | confidence {confidence:.2f}",
            results=[r.to_dict() for r in results], **meta,
        )
        return results, meta

    # ------------------------------------------------------------------
    def _parse(self, data: Dict[str, Any], candidates: List[CategoryScore],
               findings: List[SpecialistFinding]) -> List[IntentionResult]:
        raw = data.get("final_intentions") or data.get("intentions") or []
        if isinstance(raw, dict):
            raw = [{"category": k, "confidence": v} for k, v in raw.items()]

        confirmed = {f.category: f for f in findings if f.confirmed}
        results: Dict[str, IntentionResult] = {}

        for item in raw:
            if isinstance(item, str):
                category, confidence, evidence, explanation = (
                    Intention.coerce(item), 0.7, [], "")
            elif isinstance(item, dict):
                category = Intention.coerce(item.get("category") or item.get("name"))
                confidence = as_float(item.get("confidence"), 0.6)
                evidence = as_list(item.get("evidence"))
                explanation = str(item.get("explanation", ""))[:400]
            else:
                continue
            if category is None:
                continue
            key = category.value
            if key in confirmed and not evidence:
                evidence = confirmed[key].evidence
            current = results.get(key)
            if current is None or confidence > current.confidence:
                results[key] = IntentionResult(key, confidence, evidence[:8],
                                               explanation)

        # Fallbacks preserve the paper's |T| >= 1 guarantee.
        if not results:
            for finding in sorted(findings, key=lambda f: f.confidence, reverse=True):
                if finding.confirmed:
                    results[finding.category] = IntentionResult(
                        finding.category, finding.confidence, finding.evidence,
                        finding.reasoning[:300])
        if not results and candidates:
            best = max(candidates, key=lambda c: c.score)
            results[best.category] = IntentionResult(best.category, best.score,
                                                     best.evidence, "")

        return sorted(results.values(), key=lambda r: r.confidence, reverse=True)[:3]

    @staticmethod
    def _fmt_candidates(candidates: List[CategoryScore]) -> str:
        if not candidates:
            return "(no candidate category was nominated)"
        return "\n".join(
            f"- {c.category} (score {c.score:.2f}) evidence: "
            f"{'; '.join(c.evidence[:3]) or 'n/a'}" for c in candidates)

    @staticmethod
    def _fmt_findings(findings: List[SpecialistFinding]) -> str:
        if not findings:
            return "(no specialist was activated)"
        return "\n".join(
            f"- {f.category}: {'CONFIRMED' if f.confirmed else 'REJECTED'} "
            f"(confidence {f.confidence:.2f})\n"
            f"    evidence: {'; '.join(f.evidence[:4]) or 'n/a'}\n"
            f"    reasoning: {f.reasoning[:300]}" for f in findings)
