from typing import Any

from .schema import AnalysisResult


INTENT_DISPLAY_NAMES = {
    "credential_theft": "Credential Theft",
    "financial_fraud": "Financial Fraud",
    "malware_distribution": "Malware Distribution",
    "personal_information_harvesting": (
        "Personal Information Harvesting"
    ),
}


INTENT_DESCRIPTIONS = {
    "credential_theft": (
        "The webpage appears to request authentication information "
        "such as an email address, username, password, PIN, OTP, "
        "security answer, or account-recovery code."
    ),
    "financial_fraud": (
        "The webpage appears to request payment or financial details "
        "such as card information, bank details, direct payment, "
        "money transfer, or investment-related information."
    ),
    "malware_distribution": (
        "The webpage appears to encourage downloading or installing "
        "a file, application, browser component, document, update, "
        "or executable program."
    ),
    "personal_information_harvesting": (
        "The webpage appears to request personal information beyond "
        "normal login credentials, such as a telephone number, "
        "address, date of birth, government ID, employment details, "
        "or other identity information."
    ),
}


INTENT_ICONS = {
    "credential_theft": "🔑",
    "financial_fraud": "💳",
    "malware_distribution": "📥",
    "personal_information_harvesting": "🪪",
}


def confidence_percentage(value: float) -> str:
    """
    Convert a confidence value from 0-1 into a percentage.
    """
    value = max(0.0, min(1.0, float(value)))
    return f"{value * 100:.1f}%"


def confidence_level(value: float) -> str:
    """
    Convert numerical confidence into a friendly confidence level.
    """
    value = float(value)

    if value >= 0.85:
        return "Very high"

    if value >= 0.70:
        return "High"

    if value >= 0.55:
        return "Moderate"

    if value >= 0.40:
        return "Low"

    return "Very low"


def confidence_icon(value: float) -> str:
    """
    Return a visual indicator for the confidence value.
    """
    value = float(value)

    if value >= 0.70:
        return "🟢"

    if value >= 0.55:
        return "🟡"

    return "🔴"


def display_intent(intent: str) -> str:
    """
    Convert an internal intention identifier into a readable name.
    """
    return INTENT_DISPLAY_NAMES.get(
        intent,
        intent.replace("_", " ").title(),
    )


def display_agent(agent: str) -> str:
    """
    Convert an internal agent identifier into a readable name.
    """
    replacements = {
        "vision_analysis": "Vision Analysis Agent",
        "context_enrichment": "Context Enrichment Agent",
        "classification": "Initial Classification Agent",
        "credential_theft_specialist": (
            "Credential Theft Specialist"
        ),
        "financial_fraud_specialist": (
            "Financial Fraud Specialist"
        ),
        "malware_distribution_specialist": (
            "Malware Distribution Specialist"
        ),
        "personal_information_harvesting_specialist": (
            "Personal Information Specialist"
        ),
        "validator": "Validation and Synthesis Agent",
        "validator_feedback": (
            "Validation and Synthesis Agent, Feedback Pass"
        ),
        "single_agent": "Single-Agent Baseline",
    }

    if agent in replacements:
        return replacements[agent]

    if agent.endswith("_specialist_feedback"):
        intent = agent.removesuffix("_specialist_feedback")

        return (
            f"{display_intent(intent)} Specialist, "
            "Feedback Pass"
        )

    return agent.replace("_", " ").title()


def get_intent_confidence(
    result: AnalysisResult,
    intent: str,
) -> float:
    """
    Obtain the candidate confidence for one detected intention.

    If no individual candidate confidence is available, use the final
    validation confidence.
    """
    for candidate in result.candidates:
        if candidate.intent == intent:
            return float(candidate.confidence)

    specialist_reports = result.trace.get(
        "specialists",
        {},
    )

    specialist_report = specialist_reports.get(intent)

    if isinstance(specialist_report, dict):
        confidence = specialist_report.get("confidence")

        if confidence is not None:
            try:
                return float(confidence)
            except (TypeError, ValueError):
                pass

    return float(result.confidence)


def get_intent_evidence(
    result: AnalysisResult,
    intent: str,
) -> list[str]:
    """
    Retrieve observable evidence for a detected intention.
    """
    evidence = result.evidence.get(intent, [])

    if isinstance(evidence, str):
        evidence = [evidence]

    if not evidence:
        for candidate in result.candidates:
            if candidate.intent == intent:
                evidence = candidate.evidence
                break

    cleaned = []

    for item in evidence:
        text = str(item).strip()

        if text and text not in cleaned:
            cleaned.append(text)

    return cleaned


def human_summary(
    result: AnalysisResult,
    threshold: float = 0.55,
) -> dict[str, Any]:
    """
    Create a structured human-friendly summary.

    This can be used by Streamlit, an API, or another user interface.
    """
    detected_intentions = []

    for intent in result.labels:
        detected_intentions.append(
            {
                "id": intent,
                "name": display_intent(intent),
                "icon": INTENT_ICONS.get(intent, "⚠️"),
                "description": INTENT_DESCRIPTIONS.get(
                    intent,
                    "",
                ),
                "confidence": get_intent_confidence(
                    result,
                    intent,
                ),
                "evidence": get_intent_evidence(
                    result,
                    intent,
                ),
            }
        )

    return {
        "sample_id": result.sample_id,
        "detected": bool(result.labels),
        "detected_intentions": detected_intentions,
        "number_of_intentions": len(result.labels),
        "overall_confidence": float(result.confidence),
        "confidence_level": confidence_level(
            result.confidence
        ),
        "threshold": float(threshold),
        "passed_threshold": (
            float(result.confidence) >= float(threshold)
        ),
        "evidence_consistency": float(
            result.evidence_consistency
        ),
        "agents_invoked": [
            display_agent(agent)
            for agent in result.agents_invoked
        ],
        "processing_mode": result.trace.get(
            "mode",
            "unknown",
        ),
    }


def human_text_report(
    result: AnalysisResult,
    threshold: float = 0.55,
) -> str:
    """
    Generate a plain-text human-readable report.

    Useful for terminal output, log files, reports, and exports.
    """
    summary = human_summary(
        result=result,
        threshold=threshold,
    )

    lines = [
        "=" * 72,
        "PHISHING INTENTION ANALYSIS",
        "=" * 72,
        f"Sample ID: {summary['sample_id']}",
        (
            "Processing mode: "
            f"{summary['processing_mode'].replace('_', ' ').title()}"
        ),
        (
            "Overall confidence: "
            f"{confidence_percentage(summary['overall_confidence'])} "
            f"({summary['confidence_level']})"
        ),
        (
            "Evidence consistency: "
            f"{confidence_percentage(summary['evidence_consistency'])}"
        ),
        (
            "Confidence threshold: "
            f"{confidence_percentage(summary['threshold'])}"
        ),
        (
            "Threshold result: "
            + (
                "Passed"
                if summary["passed_threshold"]
                else "Below threshold"
            )
        ),
        "",
    ]

    if not summary["detected"]:
        lines.extend(
            [
                "RESULT",
                "-" * 72,
                (
                    "No supported phishing intention was identified "
                    "from the visible screenshot evidence."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "DETECTED INTENTIONS",
                "-" * 72,
            ]
        )

        for index, intention in enumerate(
            summary["detected_intentions"],
            start=1,
        ):
            lines.extend(
                [
                    (
                        f"{index}. {intention['name']} "
                        f"({confidence_percentage(intention['confidence'])})"
                    ),
                    f"   {intention['description']}",
                    "   Supporting evidence:",
                ]
            )

            if intention["evidence"]:
                for evidence in intention["evidence"]:
                    lines.append(f"   - {evidence}")
            else:
                lines.append(
                    "   - No individual evidence item was returned."
                )

            lines.append("")

    lines.extend(
        [
            "AGENTS INVOKED",
            "-" * 72,
        ]
    )

    for agent in summary["agents_invoked"]:
        lines.append(f"- {agent}")

    lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "-" * 72,
            (
                "The output describes intentions supported by visible "
                "evidence in the supplied screenshot. The confidence "
                "score is generated by the selected vision-language "
                "model and should not be interpreted as a calibrated "
                "probability without separate calibration experiments."
            ),
            "=" * 72,
        ]
    )

    return "\n".join(lines)