"""Local Ollama annotation components."""

from .schemas import (
    AnnotationOutput,
    LabelDecision,
)

from .ollama_annotator import (
    OllamaAnnotator,
    LocalAnnotationOutput,
)

from .ollama_adjudicator import (
    OllamaAdjudicator,
    OllamaAdjudicationOutput,
)

__all__ = [
    "AnnotationOutput",
    "LabelDecision",
    "OllamaAnnotator",
    "LocalAnnotationOutput",
    "OllamaAdjudicator",
    "OllamaAdjudicationOutput",
]
