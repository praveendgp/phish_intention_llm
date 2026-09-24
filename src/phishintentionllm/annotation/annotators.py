"""Annotator agents - the MANIFEST stage.

These agents do **not** belong to the five-layer framework. Their sole job is to
produce the **manifest file**: the reference label set that the framework's
predictions are later scored against.

    Annotator A (VLM) ─┐
                       ├─→ agreement analysis ─→ Tie-Breaker (VLM, on dispute)
    Annotator B (VLM) ─┘                                 │
                                                          ▼
                                                  Finaliser (VLM)
                                                          │
                                                          ▼
                                              one ManifestRecord

Each annotator looks at the raw screenshot with no framework output in its
prompt, so the manifest stays independent of the system being evaluated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..agents.base import CATEGORY_BLOCK, VisionAgent
from ..schemas import AgentStep, AnnotatorVote, Intention
from ..utils.json_parse import as_bool, as_float, as_list

# ---------------------------------------------------------------------------
# 1. Independent annotator
# ---------------------------------------------------------------------------
ANNOTATOR_SYSTEM = (
    "You are an independent phishing-intention annotator with several years of "
    "incident-response experience. You are building a labelled reference "
    "dataset. You judge each screenshot entirely on its own merits, you are not "
    "influenced by any other analyst, and you reply with one valid JSON object."
)

ANNOTATOR_PROMPT = """You are labelling a website screenshot for a phishing
intention dataset. Your label becomes reference data, so be precise and
conservative.

{categories}

## Knowledge-base reference for the four categories
{knowledge}

## Task
Study the screenshot and decide which intention(s) it exhibits.

Return JSON:
{{
  "intentions": [
    {{"category": "<exact category name>",
      "confidence": <0.0-1.0>,
      "evidence": ["<specific element visible in the image that proves it>"]}}
  ],
  "sector": "<one of: {sectors}>",
  "is_phishing": true/false,
  "phishing_confidence": <0.0-1.0>,
  "rationale": "<2-3 sentences explaining your labelling>"
}}

Labelling rules:
- List between 1 and 3 intentions, ordered by confidence.
- Every intention needs at least one concrete element you can SEE in the image.
  Never label on inference. If your evidence contains the words "implies",
  "may", "likely", "suggests", "could" or "probably", you do NOT have evidence
  - drop that category.
- Credential Theft: a password, PIN, OTP or security-answer field is visible.
- Personal Information Harvesting requires TWO OR MORE visible input fields
  collecting identity data beyond the account identifier, OR one strong
  identifier (national ID, SSN, Aadhaar, passport, driving licence, tax ID,
  full postal address, date of birth, or an ID-document upload).
  * An email address or username is an ACCOUNT IDENTIFIER, never PII.
    "Email or mobile number" as a single sign-in field is Credential Theft ONLY.
  * A phone number alone is NOT sufficient.
  * Do NOT label PIH because a page says "verify your identity", shows a
    Terms/Privacy/Contact link, or because more fields might appear later.
  * A plain email + password login form is Credential Theft ONLY.
- Financial Fraud: card number, CVV, expiry, an amount to pay, or transfer /
  wallet details are visible.
- Malware Distribution: a download, install, update or run action is visible.
- If the page looks legitimate, set is_phishing false and return an empty list.
"""


class AnnotatorAgent(VisionAgent):
    """One independent vision annotator. Instantiate twice with different roles."""

    layer = "Manifest - Independent Annotation"
    stage = "manifest"

    def __init__(self, *args, model_role: str = "annotator_a", **kwargs):
        self.annotator_id = model_role
        letter = model_role.rsplit("_", 1)[-1].upper()
        self.name = f"Annotator {letter}"
        super().__init__(*args, model_role=model_role, **kwargs)

    # ------------------------------------------------------------------
    def annotate(self, image_b64: str,
                 step: Optional[AgentStep] = None) -> AnnotatorVote:
        step = step or self.new_step()

        knowledge_lines = []
        for intention in Intention.all():
            block = self.retriever.kb.retrieve_specialist_knowledge(intention.value)
            indicators = block.get("I_c_indicators", [])[:3]
            knowledge_lines.append(
                f"{intention.value}: {block.get('definition', '')}\n"
                + "\n".join(f"   · {i}" for i in indicators))

        prompt = ANNOTATOR_PROMPT.format(
            categories=CATEGORY_BLOCK,
            knowledge="\n".join(knowledge_lines),
            sectors=", ".join(self.cfg.sectors),
        )

        data: Dict[str, Any] = self.run_guarded(
            step, self.ask_json, prompt, image_b64, system=ANNOTATOR_SYSTEM,
            default={"intentions": []},
        )

        categories, confidences, evidence = self._parse_intentions(data)
        vote = AnnotatorVote(
            annotator=self.name,
            model=self.spec.model,
            categories=categories,
            confidences=confidences,
            evidence=evidence,
            sector=self._sector(data.get("sector")),
            is_phishing=as_bool(data.get("is_phishing"), True),
            phishing_confidence=as_float(data.get("phishing_confidence"), 0.5),
            rationale=str(data.get("rationale", ""))[:1200],
            latency=float(data.get("_meta", {}).get("latency", 0.0) or 0.0),
        )

        step.finish(
            "Labelled: " + (", ".join(f"{Intention.short(c)} {confidences.get(c, 0):.2f}"
                                      for c in categories) or "no intention"),
            vote=vote.to_dict(),
        )
        return vote

    # ------------------------------------------------------------------
    def _sector(self, value: Any) -> str:
        text = str(value or "other").strip().lower()
        for sector in self.cfg.sectors:
            if sector.lower() == text or sector.lower() in text:
                return sector
        return "other"

    def _parse_intentions(self, data: Dict[str, Any]
                          ) -> Tuple[List[str], Dict[str, float], Dict[str, List[str]]]:
        categories: List[str] = []
        confidences: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}

        raw = data.get("intentions") or data.get("categories") or []
        if isinstance(raw, dict):
            raw = [{"category": k, "confidence": v} for k, v in raw.items()]

        for item in raw:
            if isinstance(item, str):
                category, confidence, ev = Intention.coerce(item), 0.7, []
            elif isinstance(item, dict):
                category = Intention.coerce(
                    item.get("category") or item.get("intention") or item.get("name"))
                confidence = as_float(item.get("confidence") or item.get("score"), 0.6)
                ev = as_list(item.get("evidence"))
            else:
                continue
            if category is None:
                continue
            key = category.value
            if key not in categories:
                categories.append(key)
                confidences[key] = confidence
                evidence[key] = ev[:6]
            else:
                confidences[key] = max(confidences[key], confidence)
                evidence[key] = (evidence[key] + ev)[:6]

        ordered = sorted(categories, key=lambda c: confidences.get(c, 0), reverse=True)
        return ordered[:3], confidences, evidence


# ---------------------------------------------------------------------------
# 2. Agreement analysis (pure function - unit tested)
# ---------------------------------------------------------------------------
def analyse_agreement(votes: List[AnnotatorVote],
                      margin: float = 0.15) -> Dict[str, Any]:
    """Compare two independent annotator votes and decide if a tie-break is due."""
    usable = [v for v in votes if v.error is None]
    if len(usable) < 2:
        only = usable[0].categories if usable else []
        return {"status": "single annotator", "agreed": sorted(only),
                "disputed": [], "requires_tiebreak": False,
                "tiebreak_categories": [], "union": sorted(only)}

    a, b = usable[0], usable[1]
    set_a, set_b = set(a.categories), set(b.categories)
    agreed = sorted(set_a & set_b)
    disputed = sorted(set_a ^ set_b)

    # A category claimed by only one annotator is escalated unless that
    # annotator is decisively confident.
    solo_threshold = max(0.5, 1.0 - margin)
    requires = [c for c in disputed
                if (a if c in set_a else b).confidences.get(c, 0.0) < solo_threshold]

    if not disputed:
        status = "full agreement"
    elif agreed:
        status = "partial agreement"
    else:
        status = "conflict"

    return {"status": status, "agreed": agreed, "disputed": disputed,
            "requires_tiebreak": bool(requires), "tiebreak_categories": requires,
            "union": sorted(set_a | set_b)}


def format_vote(vote: AnnotatorVote) -> str:
    if vote.error:
        return f"{vote.annotator} ({vote.model}): FAILED - {vote.error}"
    if not vote.categories:
        return f"{vote.annotator} ({vote.model}): no intention identified."
    items = ", ".join(f"{c} [{vote.confidences.get(c, 0):.2f}]"
                      for c in vote.categories)
    return (f"{vote.annotator} ({vote.model}): {items}\n"
            f"   evidence: {'; '.join(sum(vote.evidence.values(), [])[:4]) or 'n/a'}\n"
            f"   rationale: {vote.rationale[:350]}")


# ---------------------------------------------------------------------------
# 3. Tie-Breaker
# ---------------------------------------------------------------------------
TIEBREAK_SYSTEM = (
    "You are the deciding adjudicator for a phishing-intention labelling "
    "dispute in a reference dataset. Two annotators disagreed. You inspect the "
    "screenshot yourself and rule on each disputed category. Reply with one "
    "valid JSON object."
)

TIEBREAK_PROMPT = """Two annotators disagree about this screenshot.

## Their votes
{votes}

## Disputed categories requiring your ruling
{disputed}

## Decisive indicators from the knowledge base
{knowledge}

{categories}

## Task
Look at the screenshot and rule on EACH disputed category independently.

Return JSON:
{{
  "decisions": {{"<category name>": true/false}},
  "confidences": {{"<category name>": <0.0-1.0>}},
  "evidence": {{"<category name>": ["<what you can actually see>"]}},
  "reasoning": "<2-3 sentences justifying the rulings>"
}}

Rule true only if you can point to a visible element that proves the intention.

EVIDENCE STANDARD - applies to every ruling you make:
- Rule a category TRUE only if a specific input field or UI element visible in
  the image proves it. Never rule on inference.
- Reject any claim whose justification contains "implies", "may", "likely",
  "suggests", "could" or "probably" - that is not evidence.
- Personal Information Harvesting specifically: an email address or username is
  an ACCOUNT IDENTIFIER, not personal information. A plain email + password
  login form is Credential Theft ONLY. Require two or more visible identity
  fields (name, DOB, address, phone, postcode, occupation) or one strong
  identifier (national ID, passport, licence, tax ID, ID-document upload).
- Do not add a category that neither annotator proposed unless the image shows
  unmistakable evidence for it.
  """


class TieBreakerAgent(VisionAgent):
    name = "Tie-Breaker"
    layer = "Manifest - Tie-Break"
    stage = "manifest"
    role = "manifest_tiebreaker"

    def resolve(self, votes: List[AnnotatorVote], disputed: List[str],
                image_b64: str, step: Optional[AgentStep] = None) -> Dict[str, Any]:
        step = step or self.new_step()

        knowledge = "\n".join(
            self.retriever.as_context(category, top_k=3, category=category)
            for category in disputed)

        prompt = TIEBREAK_PROMPT.format(
            votes="\n".join(format_vote(v) for v in votes),
            disputed="\n".join(f"- {c}" for c in disputed),
            knowledge=knowledge,
            categories=CATEGORY_BLOCK,
        )

        data = self.run_guarded(step, self.ask_json, prompt, image_b64,
                                system=TIEBREAK_SYSTEM, default={"decisions": {}})

        decisions: Dict[str, bool] = {}
        confidences: Dict[str, float] = {}
        raw_decisions = data.get("decisions") or {}
        raw_conf = data.get("confidences") or {}

        if isinstance(raw_decisions, list):     # tolerate list-of-objects form
            raw_decisions = {str(d.get("category")): d.get("decision", d.get("keep"))
                             for d in raw_decisions if isinstance(d, dict)}

        for key, value in (raw_decisions or {}).items():
            category = Intention.coerce(key)
            if category is None:
                continue
            keep = as_bool(value)
            decisions[category.value] = keep
            confidences[category.value] = as_float(
                raw_conf.get(key) if isinstance(raw_conf, dict) else None,
                0.7 if keep else 0.3)

        for category in disputed:               # anything unruled is rejected
            decisions.setdefault(category, False)
            confidences.setdefault(category, 0.3)

        result = {
            "decisions": decisions,
            "confidences": confidences,
            "evidence": data.get("evidence", {}) if isinstance(
                data.get("evidence"), dict) else {},
            "reasoning": str(data.get("reasoning", ""))[:800],
            "model": self.spec.model,
        }
        kept = [c for c, keep in decisions.items() if keep]
        step.finish(
            f"Ruled on {len(decisions)} disputed categor(ies); kept: "
            + (", ".join(Intention.short(c) for c in kept) or "none"),
            **result)
        return result


# ---------------------------------------------------------------------------
# 4. Finaliser
# ---------------------------------------------------------------------------
FINALIZER_SYSTEM = (
    "You are the senior reviewing analyst signing off a reference dataset. Two "
    "independent annotators labelled a screenshot, and a tie-breaker may have "
    "ruled on disputes. You issue the authoritative label set that will be used "
    "as ground truth. Reply with one valid JSON object."
)

FINALIZER_PROMPT = """## Independent annotator votes
{votes}

## Agreement analysis
Status: {status}
Agreed categories: {agreed}
Disputed categories: {disputed}
{tiebreak_block}

## Knowledge-base reference
{knowledge}

{categories}

## Task
Look at the screenshot and issue the final manifest label set for this sample.

Return JSON:
{{
  "labels": [
    {{"category": "<exact category name>",
      "confidence": <0.0-1.0>,
      "evidence": ["<visible proof>"],
      "source": "<both | annotator-a | annotator-b | tie-breaker | reviewer>"}}
  ],
  "rejected": [{{"category": "<name>", "reason": "<why it was dropped>"}}],
  "sector": "<final sector>",
  "is_phishing": true/false,
  "label_quality": "<high | medium | low>",
  "needs_human_review": true/false,
  "summary": "<2-3 sentence justification of the final label set>"
}}

Rules:
- Keep a category only if the visible evidence genuinely supports it.
- Categories both annotators agreed on start with a strong prior.
- Respect the tie-breaker's ruling on disputed categories unless the image
  plainly contradicts it.
- Flag needs_human_review when the evidence is weak, the page is unreadable or
  the annotators conflicted badly.
- Return at least one label for a phishing page; at most three.

EVIDENCE STANDARD - applies to every ruling you make:
- Rule a category TRUE only if a specific input field or UI element visible in
  the image proves it. Never rule on inference.
- Reject any claim whose justification contains "implies", "may", "likely",
  "suggests", "could" or "probably" - that is not evidence.
- Personal Information Harvesting specifically: an email address or username is
  an ACCOUNT IDENTIFIER, not personal information. A plain email + password
  login form is Credential Theft ONLY. Require two or more visible identity
  fields (name, DOB, address, phone, postcode, occupation) or one strong
  identifier (national ID, passport, licence, tax ID, ID-document upload).
- Do not add a category that neither annotator proposed unless the image shows
  unmistakable evidence for it."""


class FinalizerAgent(VisionAgent):
    name = "Manifest Finaliser"
    layer = "Manifest - Finalisation"
    stage = "manifest"
    role = "manifest_finalizer"

    def finalize(
        self,
        votes: List[AnnotatorVote],
        agreement: Dict[str, Any],
        image_b64: str,
        tiebreak: Optional[Dict[str, Any]] = None,
        step: Optional[AgentStep] = None,
    ) -> Dict[str, Any]:
        step = step or self.new_step()

        tiebreak_block = ""
        if tiebreak:
            decided = ", ".join(f"{k}: {'KEEP' if v else 'DROP'}"
                                for k, v in (tiebreak.get("decisions") or {}).items())
            tiebreak_block = (
                f"Tie-breaker ruling ({tiebreak.get('model', 'n/a')}): {decided}\n"
                f"   reasoning: {str(tiebreak.get('reasoning', ''))[:400]}")

        query = " ".join(agreement.get("union", [])) or "phishing page"
        prompt = FINALIZER_PROMPT.format(
            votes="\n".join(format_vote(v) for v in votes),
            status=agreement.get("status", "n/a"),
            agreed=", ".join(agreement.get("agreed", [])) or "(none)",
            disputed=", ".join(agreement.get("disputed", [])) or "(none)",
            tiebreak_block=tiebreak_block,
            knowledge=self.retriever.as_context(query, top_k=5),
            categories=CATEGORY_BLOCK,
        )

        data = self.run_guarded(step, self.ask_json, prompt, image_b64,
                                system=FINALIZER_SYSTEM, default={"labels": []})

        labels, confidences, evidence = self._parse(data, votes, agreement, tiebreak)
        payload = {
            "labels": labels,
            "confidences": confidences,
            "evidence": evidence,
            "rejected": data.get("rejected", []),
            "sector": str(data.get("sector") or
                          (votes[0].sector if votes else "other")),
            "is_phishing": as_bool(data.get("is_phishing"), True),
            "label_quality": str(data.get("label_quality", "medium")).lower(),
            "needs_human_review": as_bool(data.get("needs_human_review")),
            "summary": str(data.get("summary", ""))[:800],
        }
        # `summary` is a positional parameter of AgentStep.finish, so the
        # finaliser's own summary is stored under a distinct key.
        output = {k: v for k, v in payload.items() if k not in ("evidence", "summary")}
        output["finalizer_summary"] = payload["summary"]
        step.finish(
            "Manifest labels: " + (", ".join(Intention.short(c) for c in labels)
                                   or "none")
            + f" (quality: {payload['label_quality']})",
            **output)
        return payload

    # ------------------------------------------------------------------
    def _parse(self, data: Dict[str, Any], votes: List[AnnotatorVote],
               agreement: Dict[str, Any], tiebreak: Optional[Dict[str, Any]]):
        raw = data.get("labels") or data.get("final_intentions") or []
        if isinstance(raw, dict):
            raw = [{"category": k, "confidence": v} for k, v in raw.items()]

        confidences: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}
        order: List[str] = []

        for item in raw:
            if isinstance(item, str):
                category, confidence, ev = Intention.coerce(item), 0.7, []
            elif isinstance(item, dict):
                category = Intention.coerce(item.get("category") or item.get("name"))
                confidence = as_float(item.get("confidence"), 0.6)
                ev = as_list(item.get("evidence"))
            else:
                continue
            if category is None:
                continue
            key = category.value
            if key not in order:
                order.append(key)
            confidences[key] = max(confidences.get(key, 0.0), confidence)
            evidence[key] = (evidence.get(key, []) + ev)[:6]

        # Fallback chain: tie-break ruling -> agreed set -> union of votes.
        if not order:
            if tiebreak:
                order = [c for c, keep in (tiebreak.get("decisions") or {}).items()
                         if keep]
                confidences.update({c: tiebreak.get("confidences", {}).get(c, 0.6)
                                    for c in order})
            if not order:
                order = list(agreement.get("agreed") or agreement.get("union") or [])
                for category in order:
                    confidences.setdefault(category, max(
                        (v.confidences.get(category, 0.0) for v in votes), default=0.5))

        ordered = sorted(order, key=lambda c: confidences.get(c, 0), reverse=True)[:3]
        return ordered, {c: confidences.get(c, 0.0) for c in ordered}, \
            {c: evidence.get(c, []) for c in ordered}
