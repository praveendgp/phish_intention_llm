from pathlib import Path
from typing import Any

from . import prompts
from .schema import (
    AnalysisResult,
    Candidate,
    INTENTS,
    VisionEvidence,
    normalise_confidence,
)


class PhishIntentionPipeline:
    def __init__(
        self,
        llm,
        kb,
        threshold: float = 0.55,
        top_k: int = 3,
    ):
        self.llm = llm
        self.kb = kb
        self.threshold = threshold
        self.top_k = top_k

    def _normalise_labels(
        self,
        value: Any,
    ) -> list:
        """
        Convert different model label formats into valid intention IDs.
        """
        if value is None:
            return []

        if isinstance(value, str):
            value = [value]

        if isinstance(value, dict):
            value = list(value.keys())

        if not isinstance(value, list):
            return []

        labels = []

        for item in value:
            if isinstance(item, dict):
                item = (
                    item.get("intent")
                    or item.get("label")
                    or item.get("name")
                )

            if not isinstance(item, str):
                continue

            normalised = (
                item.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            aliases = {
                "credentials_theft": "credential_theft",
                "credential_stealing": "credential_theft",
                "financial_scam": "financial_fraud",
                "malware": "malware_distribution",
                "personal_information": (
                    "personal_information_harvesting"
                ),
                "personal_data_harvesting": (
                    "personal_information_harvesting"
                ),
                "pii_harvesting": (
                    "personal_information_harvesting"
                ),
            }

            normalised = aliases.get(
                normalised,
                normalised,
            )

            if (
                normalised in INTENTS
                and normalised not in labels
            ):
                labels.append(normalised)

        return labels

    def run(
        self,
        image_path,
        mode: str = "gated",
        sample_id: str | None = None,
    ) -> AnalysisResult:
        sample_id = (
            sample_id
            or Path(image_path).stem
        )

        agents = []

        if mode == "single":
            output = self.llm.ask(
                prompts.baseline(),
                image_path,
            )

            labels = self._normalise_labels(
                output.get("labels")
            )

            confidence = normalise_confidence(
                output.get("confidence"),
                default=0.0,
            )

            return AnalysisResult(
                sample_id=sample_id,
                labels=labels,
                confidence=confidence,
                evidence=output.get(
                    "evidence",
                    {},
                ),
                candidates=[],
                agents_invoked=[
                    "single_agent",
                ],
                evidence_consistency=(
                    1.0 if labels else 0.0
                ),
                trace={
                    "mode": mode,
                    "raw_single_agent": output,
                },
            )

        # Layer 1: vision analysis
        raw_vision = self.llm.ask(
            prompts.vision(),
            image_path,
        )

        vision_evidence = (
            VisionEvidence.model_validate(
                raw_vision
            )
        )

        agents.append(
            "vision_analysis"
        )

        # Layer 2: context enrichment
        retrieval_query = " ".join(
            [
                vision_evidence.visible_text,
                *vision_evidence.ui_elements,
                *vision_evidence.visual_cues,
                *vision_evidence.brand_signals,
            ]
        )

        common_knowledge = self.kb.search(
            retrieval_query,
            k=4,
        )

        agents.append(
            "context_enrichment"
        )

        # Layer 3: initial classification
        raw_classification = self.llm.ask(
            prompts.classify(
                vision_evidence.model_dump(),
                common_knowledge,
            )
        )

        agents.append(
            "classification"
        )

        candidates = []

        raw_candidates = raw_classification.get(
            "candidates",
            [],
        )

        if isinstance(raw_candidates, dict):
            raw_candidates = [
                raw_candidates
            ]

        if not isinstance(raw_candidates, list):
            raw_candidates = []

        for raw_candidate in raw_candidates:
            try:
                candidate = Candidate.model_validate(
                    raw_candidate
                )

                candidates.append(candidate)

            except Exception:
                continue

        candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.confidence
            ),
            reverse=True,
        )[:self.top_k]

        if mode == "always":
            active_intentions = list(INTENTS)
        else:
            active_intentions = [
                candidate.intent
                for candidate in candidates
                if candidate.confidence
                >= self.threshold
            ]

        if (
            not active_intentions
            and candidates
        ):
            active_intentions = [
                candidates[0].intent
            ]

        # Layer 4: specialist analysis
        specialist_reports = {}

        for intention in active_intentions:
            specialist_knowledge = (
                self.kb.search(
                    retrieval_query,
                    k=4,
                    category=intention,
                )
            )

            specialist_report = self.llm.ask(
                prompts.specialist(
                    intention,
                    vision_evidence.model_dump(),
                    specialist_knowledge,
                )
            )

            # Normalise specialist confidence immediately.
            specialist_report[
                "confidence"
            ] = normalise_confidence(
                specialist_report.get(
                    "confidence"
                ),
                default=0.0,
            )

            specialist_reports[
                intention
            ] = specialist_report

            agents.append(
                f"{intention}_specialist"
            )

        # Layer 5: validation and synthesis
        validation_output = self.llm.ask(
            prompts.validate(
                vision_evidence.model_dump(),
                [
                    candidate.model_dump()
                    for candidate in candidates
                ],
                specialist_reports,
            )
        )

        agents.append(
            "validator"
        )

        confidence = normalise_confidence(
            validation_output.get(
                "confidence"
            ),
            default=0.0,
        )

        # Feedback loop
        if (
            confidence < self.threshold
            and mode == "gated"
        ):
            for intention in INTENTS:
                if intention in specialist_reports:
                    continue

                specialist_knowledge = (
                    self.kb.search(
                        retrieval_query,
                        k=4,
                        category=intention,
                    )
                )

                specialist_report = self.llm.ask(
                    prompts.specialist(
                        intention,
                        vision_evidence.model_dump(),
                        specialist_knowledge,
                    )
                )

                specialist_report[
                    "confidence"
                ] = normalise_confidence(
                    specialist_report.get(
                        "confidence"
                    ),
                    default=0.0,
                )

                specialist_reports[
                    intention
                ] = specialist_report

                agents.append(
                    (
                        f"{intention}"
                        "_specialist_feedback"
                    )
                )

            validation_output = self.llm.ask(
                prompts.validate(
                    vision_evidence.model_dump(),
                    [
                        candidate.model_dump()
                        for candidate in candidates
                    ],
                    specialist_reports,
                )
            )

            confidence = normalise_confidence(
                validation_output.get(
                    "confidence"
                ),
                default=0.0,
            )

            agents.append(
                "validator_feedback"
            )

        labels = self._normalise_labels(
            validation_output.get("labels")
        )

        evidence_consistency = (
            normalise_confidence(
                validation_output.get(
                    "evidence_consistency"
                ),
                default=0.0,
            )
        )

        return AnalysisResult(
            sample_id=sample_id,
            labels=labels,
            confidence=confidence,
            evidence=validation_output.get(
                "evidence",
                {},
            ),
            candidates=candidates,
            agents_invoked=agents,
            evidence_consistency=(
                evidence_consistency
            ),
            trace={
                "mode": mode,
                "vision": (
                    vision_evidence.model_dump()
                ),
                "raw_vision": raw_vision,
                "retrieved_common": (
                    common_knowledge
                ),
                "raw_classification": (
                    raw_classification
                ),
                "specialists": (
                    specialist_reports
                ),
                "raw_validation": (
                    validation_output
                ),
            },
        )