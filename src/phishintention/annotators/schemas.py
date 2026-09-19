from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class LabelDecision(BaseModel):
    present: bool
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=4)

class AnnotationOutput(BaseModel):
    credential_theft: LabelDecision
    financial_fraud: LabelDecision
    malware_distribution: LabelDecision
    personal_information_harvesting: LabelDecision
    image_quality: Literal["usable", "unusable"]
    exclusion_reason: str = ""
    annotation_notes: str = ""

    @model_validator(mode="after")
    def validate_quality(self):
        if self.image_quality == "unusable" and not self.exclusion_reason.strip():
            raise ValueError("exclusion_reason is required when image_quality is unusable")
        return self

    def binary_labels(self) -> dict[str, int]:
        return {
            "credential_theft": int(self.credential_theft.present),
            "financial_fraud": int(self.financial_fraud.present),
            "malware_distribution": int(self.malware_distribution.present),
            "personal_information_harvesting": int(self.personal_information_harvesting.present),
        }
