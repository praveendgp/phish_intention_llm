import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Intent = Literal[
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]


INTENTS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

def normalise_confidence(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert different model-generated confidence formats into
    one floating-point value between 0 and 1.

    Supported examples:
        0.85
        "0.85"
        "85%"
        [0.85]
        {"score": 0.85}
        {"confidence": 0.85}
        [{"score": 0.85}, {"score": 0.70}]
    """

    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        number = float(value)

        # Tolerate percentage-style numeric values such as 85.
        if number > 1 and number <= 100:
            number = number / 100

        return max(
            0.0,
            min(1.0, number),
        )

    if isinstance(value, str):
        cleaned = value.strip()

        if not cleaned:
            return default

        is_percentage = cleaned.endswith("%")

        cleaned = cleaned.replace("%", "").strip()

        try:
            number = float(cleaned)

            if is_percentage or (
                number > 1 and number <= 100
            ):
                number = number / 100

            return max(
                0.0,
                min(1.0, number),
            )

        except ValueError:
            return default

    if isinstance(value, dict):
        preferred_keys = [
            "confidence",
            "score",
            "overall_confidence",
            "probability",
            "value",
        ]

        for key in preferred_keys:
            if key in value:
                return normalise_confidence(
                    value[key],
                    default=default,
                )

        numeric_values = []

        for item in value.values():
            confidence = normalise_confidence(
                item,
                default=-1.0,
            )

            if confidence >= 0:
                numeric_values.append(confidence)

        if numeric_values:
            return max(numeric_values)

        return default

    if isinstance(value, list):
        if not value:
            return default

        confidence_values = []

        for item in value:
            confidence = normalise_confidence(
                item,
                default=-1.0,
            )

            if confidence >= 0:
                confidence_values.append(confidence)

        if not confidence_values:
            return default

        # For an overall result, use the highest supported confidence.
        return max(confidence_values)

    return default

def evidence_item_to_string(item: Any) -> str:
    """
    Convert vision-model evidence into a consistent readable string.

    Ollama vision models may return:
    - plain strings;
    - dictionaries describing UI elements;
    - lists;
    - numbers or other primitive values.

    The rest of the pipeline expects textual evidence, so all supported
    values are converted into deterministic strings.
    """

    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        readable_parts = []

        preferred_keys = [
            "type",
            "name",
            "label",
            "value",
            "text",
            "position",
            "description",
        ]

        # Put common visual attributes first.
        for key in preferred_keys:
            value = item.get(key)

            if value not in (None, "", [], {}):
                readable_parts.append(
                    f"{key}={evidence_item_to_string(value)}"
                )

        # Preserve any additional properties returned by the model.
        for key, value in item.items():
            if key in preferred_keys:
                continue

            if value not in (None, "", [], {}):
                readable_parts.append(
                    f"{key}={evidence_item_to_string(value)}"
                )

        if readable_parts:
            return ", ".join(readable_parts)

        return json.dumps(
            item,
            ensure_ascii=False,
            sort_keys=True,
        )

    if isinstance(item, list):
        converted = [
            evidence_item_to_string(value)
            for value in item
        ]

        converted = [
            value
            for value in converted
            if value
        ]

        return "; ".join(converted)

    return str(item).strip()


def normalise_evidence_list(value: Any) -> list:
    """
    Normalise a model response into a list of non-empty strings.
    """

    if value is None:
        return []

    if not isinstance(value, list):
        value = [value]

    normalised = []

    for item in value:
        text = evidence_item_to_string(item)

        if text:
            normalised.append(text)

    return normalised


class VisionEvidence(BaseModel):
    visible_text: str = ""

    ui_elements: list[str] = Field(
        default_factory=list
    )

    visual_cues: list[str] = Field(
        default_factory=list
    )

    brand_signals: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        "visible_text",
        mode="before",
    )
    @classmethod
    def normalise_visible_text(
        cls,
        value: Any,
    ) -> str:
        return evidence_item_to_string(value)

    @field_validator(
        "ui_elements",
        "visual_cues",
        "brand_signals",
        mode="before",
    )
    @classmethod
    def normalise_evidence_fields(
        cls,
        value: Any,
    ) -> list[str]:
        return normalise_evidence_list(value)


class Candidate(BaseModel):
    intent: Intent

    confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
    )

    evidence: list[str] = Field(
        default_factory=list
    )

    @field_validator(
        "confidence",
        mode="before",
    )
    @classmethod
    def normalise_candidate_confidence(
        cls,
        value: Any,
    ) -> float:
        return normalise_confidence(value)

    @field_validator(
        "evidence",
        mode="before",
    )
    @classmethod
    def normalise_candidate_evidence(
        cls,
        value: Any,
    ) -> list[str]:
        return normalise_evidence_list(value)


class AnalysisResult(BaseModel):
    sample_id: str

    labels: list[Intent]

    confidence: float = Field(
        ge=0,
        le=1,
    )

    evidence: dict[str, list[str]]

    candidates: list[Candidate]

    agents_invoked: list[str]

    evidence_consistency: float = Field(
        ge=0,
        le=1,
    )

    trace: dict = Field(
        default_factory=dict
    )

    @field_validator(
        "evidence",
        mode="before",
    )
    
    @classmethod
    def normalise_final_evidence(
        cls,
        value: Any,
    ) -> dict[str, list[str]]:
        """
        Accept both expected dictionary evidence and occasional list/string
        responses from a vision-language model.
        """

        if value is None:
            return {}

        if isinstance(value, dict):
            return {
                str(key): normalise_evidence_list(items)
                for key, items in value.items()
            }

        return {
            "general": normalise_evidence_list(value)
        }

    @field_validator(
            "confidence",
            "evidence_consistency",
            mode="before",
        )
    @classmethod
    def normalise_result_scores(
        cls,
        value: Any,
    ) -> float:
        return normalise_confidence(value)