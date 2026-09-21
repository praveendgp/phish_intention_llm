"""PhishIntentionLLM - Multi-Agent RAG for phishing intention discovery."""

from .config import load_config
from .schemas import (AnalysisResult, AnnotatorVote, Intention, IntentionResult,
                      ManifestRecord, ManualAnnotation, Sample)

__version__ = "2.0.0"
__all__ = ["load_config", "Intention", "Sample", "AnalysisResult",
           "IntentionResult", "AnnotatorVote", "ManifestRecord",
           "ManualAnnotation"]
