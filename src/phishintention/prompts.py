import json
INTENT_DEFS={
"credential_theft":"Requests authentication secrets such as username, password, PIN, OTP or recovery code.",
"financial_fraud":"Seeks payment card/bank data, direct transfer, purchase, investment or monetary exploitation.",
"malware_distribution":"Induces a file, application, update, document or executable download/install.",
"personal_information_harvesting":"Collects identity/contact/demographic/health/government identifiers beyond authentication."}
def vision():
    return """
You are a defensive website-screenshot vision analyst.

Analyse only content visibly present in the supplied website screenshot.
Do not classify the phishing intention in this step.

Return exactly this JSON structure:

{
  "visible_text": "short security-relevant visible text",
  "ui_elements": [
    "short plain-string description"
  ],
  "visual_cues": [
    "short plain-string description"
  ],
  "brand_signals": [
    "short plain-string description"
  ]
}

Field requirements:

- visible_text must be one concise string containing only the most
  security-relevant visible words.
- ui_elements must contain at most 8 short strings.
- visual_cues must contain at most 6 short strings.
- brand_signals must contain at most 4 short strings.
- Do not return dictionaries inside arrays.
- Do not reproduce the entire page verbatim.
- Do not include Markdown or code fences.
- Return one complete valid JSON object only.
""".strip()
def classify(ev,kb): return f"Act as the initial multi-label phishing-intention classifier. Labels and definitions: {json.dumps(INTENT_DEFS)}. Evidence: {json.dumps(ev)}. Retrieved patterns: {json.dumps(kb)}. Return JSON with candidates array; each item has intent, confidence 0..1, and evidence array. Return 1-3 candidates, no unsupported claims."
def specialist(intent,ev,kb): return f"You are the {intent} defensive specialist. Definition: {INTENT_DEFS[intent]}. Evidence: {json.dumps(ev)}. Retrieved specialist knowledge: {json.dumps(kb)}. Return JSON with supported boolean, confidence 0..1, and evidence array. Evidence must be observable in input."
def validate(
    evidence,
    candidates,
    specialist_reports,
):
    return f"""
You are the Validation and Synthesis Agent for a defensive phishing
intention analysis system.

Use only the supplied observable screenshot evidence.

Valid labels:

{json.dumps(INTENT_DEFS, indent=2)}

Vision evidence:

{json.dumps(evidence, indent=2)}

Initial candidates:

{json.dumps(candidates, indent=2)}

Specialist reports:

{json.dumps(specialist_reports, indent=2)}

Return exactly one JSON object with this structure:

{{
  "labels": [
    "credential_theft"
  ],
  "confidence": 0.85,
  "evidence": {{
    "credential_theft": [
      "short observable evidence statement"
    ]
  }},
  "evidence_consistency": 0.90
}}

Strict requirements:

1. labels must contain only valid intention strings.
2. confidence must be one number between 0 and 1.
3. evidence must be an object keyed by intention.
4. evidence_consistency must be one number between 0 and 1.
5. Return JSON only.
""".strip()
def baseline(): return f"Single defensive analyst. From the screenshot classify zero or more labels from {json.dumps(INTENT_DEFS)}. Return JSON with labels, confidence, and evidence object. Use observable evidence only."
