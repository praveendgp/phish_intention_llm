from pydantic import BaseModel, Field
from typing import Literal
Intent=Literal["credential_theft","financial_fraud","malware_distribution","personal_information_harvesting"]
INTENTS=["credential_theft","financial_fraud","malware_distribution","personal_information_harvesting"]
class VisionEvidence(BaseModel):
    visible_text:str=""
    ui_elements:list[str]=Field(default_factory=list)
    visual_cues:list[str]=Field(default_factory=list)
    brand_signals:list[str]=Field(default_factory=list)
class Candidate(BaseModel):
    intent:Intent
    confidence:float=Field(ge=0,le=1)
    evidence:list[str]=Field(default_factory=list)
class AnalysisResult(BaseModel):
    sample_id:str
    labels:list[Intent]
    confidence:float=Field(ge=0,le=1)
    evidence:dict[str,list[str]]
    candidates:list[Candidate]
    agents_invoked:list[str]
    evidence_consistency:float=Field(ge=0,le=1)
    trace:dict=Field(default_factory=dict)
