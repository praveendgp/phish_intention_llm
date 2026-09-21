"""Typed data structures shared across the PhishIntentionLLM project."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Intention taxonomy (Section 3 of the base paper)
# ---------------------------------------------------------------------------
class Intention(str, Enum):
    CREDENTIAL_THEFT = "Credential Theft"
    FINANCIAL_FRAUD = "Financial Fraud"
    MALWARE_DISTRIBUTION = "Malware Distribution"
    PERSONAL_INFO = "Personal Information Harvesting"

    @classmethod
    def all(cls) -> List["Intention"]:
        return [cls.CREDENTIAL_THEFT, cls.FINANCIAL_FRAUD,
                cls.MALWARE_DISTRIBUTION, cls.PERSONAL_INFO]

    @classmethod
    def short(cls, value: "Intention | str") -> str:
        mapping = {
            cls.CREDENTIAL_THEFT.value: "CT",
            cls.FINANCIAL_FRAUD.value: "FF",
            cls.MALWARE_DISTRIBUTION.value: "MD",
            cls.PERSONAL_INFO.value: "PIH",
        }
        return mapping.get(str(getattr(value, "value", value)), "??")

    @classmethod
    def coerce(cls, raw: Any) -> Optional["Intention"]:
        """Normalise free-form model output to a canonical category."""
        if raw is None:
            return None
        text = " ".join(str(raw).strip().lower()
                        .replace("_", " ").replace("-", " ").split())
        table = {
            "credential theft": cls.CREDENTIAL_THEFT,
            "credentials theft": cls.CREDENTIAL_THEFT,
            "credential harvesting": cls.CREDENTIAL_THEFT,
            "ct": cls.CREDENTIAL_THEFT,
            "financial fraud": cls.FINANCIAL_FRAUD,
            "finance fraud": cls.FINANCIAL_FRAUD,
            "payment fraud": cls.FINANCIAL_FRAUD,
            "ff": cls.FINANCIAL_FRAUD,
            "malware distribution": cls.MALWARE_DISTRIBUTION,
            "malware": cls.MALWARE_DISTRIBUTION,
            "md": cls.MALWARE_DISTRIBUTION,
            "personal information harvesting": cls.PERSONAL_INFO,
            "personal info harvesting": cls.PERSONAL_INFO,
            "personal information": cls.PERSONAL_INFO,
            "pii harvesting": cls.PERSONAL_INFO,
            "pih": cls.PERSONAL_INFO,
        }
        if text in table:
            return table[text]
        for key, value in table.items():
            if len(key) > 3 and key in text:
                return value
        return None


INTENTION_DESCRIPTIONS: Dict[str, str] = {
    Intention.CREDENTIAL_THEFT.value:
        "Counterfeit login interfaces that capture usernames, passwords, OTPs "
        "or security answers.",
    Intention.FINANCIAL_FRAUD.value:
        "Pages engineered to move money: card/CVV capture, fake invoices, bogus "
        "investments, refund and tech-support scams, crypto wallet drains.",
    Intention.MALWARE_DISTRIBUTION.value:
        "Pages that push an executable payload - fake updates, codec installs, "
        "scan alerts, forced downloads or extension installs.",
    Intention.PERSONAL_INFO.value:
        "Forms that over-collect identity data - national ID, address, DOB, "
        "phone, employment, health or KYC document uploads.",
}


# ---------------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """A screenshot from any supported dataset source."""

    sample_id: str
    source: str                       # "putra" | "phishiris"
    screenshot_path: str
    is_phishing: bool = True
    brand: Optional[str] = None
    split: Optional[str] = None
    url: Optional[str] = None
    record_dir: Optional[str] = None
    assets: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Agent bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class AgentStep:
    """One node of an agent flow - rendered live in the UI."""

    agent: str
    layer: str
    model: str
    stage: str = "framework"          # "manifest" | "framework"
    vision: bool = False              # did this agent actually receive the image?
    status: str = "pending"           # pending|running|done|skipped|error
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    summary: str = ""
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        if self.started_at and self.ended_at:
            return round(self.ended_at - self.started_at, 2)
        return 0.0

    def start(self) -> "AgentStep":
        self.status = "running"
        self.started_at = time.time()
        return self

    def finish(self, summary: str = "", **output) -> "AgentStep":
        self.status = "done"
        self.ended_at = time.time()
        self.summary = summary
        self.output.update(output)
        return self

    def fail(self, error: str) -> "AgentStep":
        self.status = "error"
        self.ended_at = time.time()
        self.error = error
        return self

    def skip(self, reason: str = "") -> "AgentStep":
        self.status = "skipped"
        self.ended_at = self.ended_at or time.time()
        self.summary = reason
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


# ---------------------------------------------------------------------------
# Framework artefacts
# ---------------------------------------------------------------------------
@dataclass
class VisualElements:
    """Layer 1 output - what the VLM actually saw."""

    ocr_text: str = ""
    interface_elements: List[str] = field(default_factory=list)
    form_fields: List[str] = field(default_factory=list)
    buttons: List[str] = field(default_factory=list)
    branding: List[str] = field(default_factory=list)
    layout: str = ""
    domain_hints: List[str] = field(default_factory=list)
    sector: str = "other"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnrichedContext:
    """Layer 2 output."""

    elements: VisualElements
    tagged_elements: List[Dict[str, str]] = field(default_factory=list)
    security_implications: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    retrieved_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": self.elements.to_dict(),
            "tagged_elements": self.tagged_elements,
            "security_implications": self.security_implications,
            "hypotheses": self.hypotheses,
            "retrieved_patterns": self.retrieved_patterns,
        }


@dataclass
class CategoryScore:
    category: str
    score: float
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpecialistFinding:
    category: str
    confirmed: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentionResult:
    category: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def short(self) -> str:
        return Intention.short(self.category)

    @property
    def risk_band(self) -> str:
        if self.confidence >= 0.85:
            return "Very High"
        if self.confidence >= 0.70:
            return "High"
        if self.confidence >= 0.55:
            return "Moderate"
        return "Low"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["short"] = self.short
        data["risk_band"] = self.risk_band
        return data


@dataclass
class AnalysisResult:
    """Framework prediction for one screenshot (the thing being evaluated)."""

    sample_id: str
    source: str
    screenshot_path: str
    intentions: List[IntentionResult] = field(default_factory=list)
    sector: str = "other"
    overall_confidence: float = 0.0
    is_phishing: bool = True
    phishing_score: float = 0.0
    verdict: str = "Unknown"
    risk_summary: str = ""
    recommended_action: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    specialist_findings: List[SpecialistFinding] = field(default_factory=list)
    category_scores: List[CategoryScore] = field(default_factory=list)
    feedback_loop_used: bool = False
    vision_calls: int = 0
    elapsed: float = 0.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def labels(self) -> List[str]:
        return [i.category for i in self.intentions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "source": self.source,
            "screenshot_path": self.screenshot_path,
            "is_phishing": self.is_phishing,
            "phishing_score": self.phishing_score,
            "verdict": self.verdict,
            "risk_summary": self.risk_summary,
            "recommended_action": self.recommended_action,
            "sector": self.sector,
            "overall_confidence": self.overall_confidence,
            "intentions": [i.to_dict() for i in self.intentions],
            "labels": self.labels,
            "specialist_findings": [s.to_dict() for s in self.specialist_findings],
            "category_scores": [c.to_dict() for c in self.category_scores],
            "steps": [s.to_dict() for s in self.steps],
            "feedback_loop_used": self.feedback_loop_used,
            "vision_calls": self.vision_calls,
            "elapsed": self.elapsed,
            "created_at": self.created_at,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Manifest artefacts (annotator stage)
# ---------------------------------------------------------------------------
@dataclass
class AnnotatorVote:
    """One independent annotator opinion on a screenshot."""

    annotator: str
    model: str
    categories: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    sector: str = "other"
    is_phishing: bool = True
    phishing_confidence: float = 0.0
    rationale: str = ""
    latency: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestRecord:
    """One line of the manifest file - the reference label for a sample.

    The manifest is produced by the annotator ensemble and is what the
    framework's predictions are later scored against.
    """

    sample_id: str
    source: str
    screenshot_path: str
    labels: List[str] = field(default_factory=list)
    confidences: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    sector: str = "other"
    is_phishing: bool = True
    brand: Optional[str] = None
    split: Optional[str] = None

    # provenance
    agreement: str = "n/a"
    agreed: List[str] = field(default_factory=list)
    disputed: List[str] = field(default_factory=list)
    tie_break_used: bool = False
    tie_break_decisions: Dict[str, bool] = field(default_factory=dict)
    annotators: List[str] = field(default_factory=list)
    votes: List[AnnotatorVote] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    finalizer_summary: str = ""
    needs_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    human_verified: bool = False

    manifest_run_id: str = ""
    elapsed: float = 0.0
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None

    @property
    def n_intentions(self) -> int:
        return len(self.labels)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "screenshot_path": self.screenshot_path,
            "labels": self.labels,
            "n_intentions": self.n_intentions,
            "confidences": self.confidences,
            "evidence": self.evidence,
            "sector": self.sector,
            "is_phishing": self.is_phishing,
            "brand": self.brand,
            "split": self.split,
            "agreement": self.agreement,
            "agreed": self.agreed,
            "disputed": self.disputed,
            "tie_break_used": self.tie_break_used,
            "tie_break_decisions": self.tie_break_decisions,
            "annotators": self.annotators,
            "votes": [v.to_dict() for v in self.votes],
            "steps": [s.to_dict() for s in self.steps],
            "finalizer_summary": self.finalizer_summary,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "human_verified": self.human_verified,
            "manifest_run_id": self.manifest_run_id,
            "elapsed": self.elapsed,
            "created_at": self.created_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManifestRecord":
        record = cls(
            sample_id=data.get("sample_id", ""),
            source=data.get("source", ""),
            screenshot_path=data.get("screenshot_path", ""),
            labels=list(data.get("labels", [])),
            confidences=dict(data.get("confidences", {})),
            evidence=dict(data.get("evidence", {})),
            sector=data.get("sector", "other"),
            is_phishing=bool(data.get("is_phishing", True)),
            brand=data.get("brand"),
            split=data.get("split"),
            agreement=data.get("agreement", "n/a"),
            agreed=list(data.get("agreed", [])),
            disputed=list(data.get("disputed", [])),
            tie_break_used=bool(data.get("tie_break_used", False)),
            tie_break_decisions=dict(data.get("tie_break_decisions", {})),
            annotators=list(data.get("annotators", [])),
            finalizer_summary=data.get("finalizer_summary", ""),
            needs_review=bool(data.get("needs_review", False)),
            review_reasons=list(data.get("review_reasons", [])),
            human_verified=bool(data.get("human_verified", False)),
            manifest_run_id=data.get("manifest_run_id", ""),
            elapsed=float(data.get("elapsed", 0.0) or 0.0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            error=data.get("error"),
        )
        record.votes = [AnnotatorVote(**{k: v for k, v in vote.items()
                                         if k in AnnotatorVote.__annotations__})
                        for vote in data.get("votes", [])]
        return record


@dataclass
class ManualAnnotation:
    """Human label produced by the manual annotation module."""

    sample_id: str
    annotator: str
    categories: List[str]
    sector: str = "other"
    notes: str = ""
    is_phishing: bool = True
    reviewed: bool = False
    reviewer: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
