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
        ge=0,
        le=1,
    )

    evidence: list[str] = Field(
        default_factory=list
    )

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