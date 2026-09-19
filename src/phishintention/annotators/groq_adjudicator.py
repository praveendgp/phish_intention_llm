from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import (
    BadRequestError,
    Groq,
)
from pydantic import BaseModel, Field

from .utils import image_data_uri


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


class FinalLabelDecision(BaseModel):
    present: bool

    confidence: float = Field(
        ge=0,
        le=1,
    )

    evidence: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

    decision_reason: str = Field(
        default="",
        max_length=500,
    )


class GroqAdjudicationOutput(BaseModel):
    credential_theft: FinalLabelDecision
    financial_fraud: FinalLabelDecision
    malware_distribution: FinalLabelDecision
    personal_information_harvesting: FinalLabelDecision

    image_quality: str = Field(
        pattern="^(usable|unusable)$"
    )

    exclusion_reason: str = Field(
        default="",
        max_length=500,
    )

    adjudication_summary: str = Field(
        default="",
        max_length=1000,
    )

    def binary_labels(
        self,
    ) -> dict[str, int]:
        return {
            "credential_theft": int(
                self.credential_theft.present
            ),
            "financial_fraud": int(
                self.financial_fraud.present
            ),
            "malware_distribution": int(
                self.malware_distribution.present
            ),
            "personal_information_harvesting": int(
                self.personal_information_harvesting.present
            ),
        }


GROQ_ADJUDICATION_PROMPT = """
You are a defensive cybersecurity annotation adjudicator.

You are given:

1. A static website screenshot.
2. An independent OpenAI annotation.
3. An independent Gemini annotation.

Resolve disagreements using only screenshot-visible evidence.
Do not automatically select either model's answer.

INTENTION DEFINITIONS

credential_theft:
Collection of authentication secrets such as username, login email,
password, PIN, OTP, security answer or account-recovery code.

financial_fraud:
Suspicious collection of payment, card, bank, transfer,
cryptocurrency, investment or other direct financial information.

malware_distribution:
A suspicious attempt to induce downloading or installing an
application, APK, executable, attachment, fake update, browser
component or another file.

personal_information_harvesting:
Suspicious collection of identity or personal information beyond
authentication, such as phone number, address, date of birth,
passport, government ID, tax ID, employment, demographic or
health information.

DECISION RULES

1. Assign every label independently.
2. Multiple labels may be present.
3. Use only screenshot-visible evidence.
4. A normal login form alone does not prove credential theft.
5. A normal checkout form alone does not prove financial fraud.
6. A download button alone does not prove malware distribution.
7. An ordinary registration or delivery form alone does not prove
   personal-information harvesting.
8. Prefer decisions supported by concrete visible evidence.
9. Mark image_quality as unusable when the screenshot is blank,
   unreadable, incomplete or only an error page.
10. Keep evidence concise.
11. Confidence measures confidence in the annotation decision,
    not the probability that the website is hishing.
12. Return only one valid JSON object matching the required schema.
13. Complete every string, list and JSON object.

OPENAI ANNOTATION

{openai_annotation}

GEMINI ANNOTATION

{gemini_annotation}

DISAGREEMENT LABELS

{disagreement_labels}
""".strip()


class GroqAdjudicator:
    def __init__(
        self,
        model: str | None = None,
        max_image_side: int = 1600,
        max_retries: int | None = None,
    ):
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add GROQ_API_KEY to the project .env file."
            )

        self.model = (
            model
            or os.getenv(
                "GROQ_ADJUDICATION_MODEL",
                "qwen/qwen3.8-27b",
            )
        )

        self.max_image_side = int(
            max_image_side
        )

        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(
                os.getenv(
                    "GROQ_MAX_RETRIES",
                    "3",
                )
            )
        )

        self.client = Groq(
            api_key=api_key
        )

    def _build_prompt(
        self,
        openai_annotation: dict,
        gemini_annotation: dict,
        disagreement_labels: list[str],
        retry: bool = False,
    ) -> str:
        prompt = (
            GROQ_ADJUDICATION_PROMPT.format(
                openai_annotation=json.dumps(
                    openai_annotation,
                    indent=2,
                    ensure_ascii=False,
                ),
                gemini_annotation=json.dumps(
                    gemini_annotation,
                    indent=2,
                    ensure_ascii=False,
                ),
                disagreement_labels=json.dumps(
                    disagreement_labels,
                    ensure_ascii=False,
                ),
            )
        )

        if retry:
            prompt += """

CORRECTION REQUIREMENTS

The preceding response was incomplete or did not match the schema.

Return the complete adjudication again.

- Return valid JSON only.
- Keep evidence statements short.
- Return no more than four evidence statements per label.
- Complete every string, array and object.
- Do not include Markdown.
"""

        return prompt

    def _make_request(
        self,
        image_path: str | Path,
        prompt: str,
    ):

        return (
            self.client
            .chat
            .completions
            .create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_uri(
                                        image_path,
                                        self.max_image_side,
                                    )
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
                max_completion_tokens=1800,
                response_format={
    "type": "json_object",
},
                stream=False,
            )
        )

    @staticmethod
    def _extract_balanced_json(
        raw_text: str,
    ) -> str:
        """
        Extract the first complete, balanced JSON object.

        This removes malformed text that Groq may append after a
        complete JSON object.
        """
        if not raw_text:
            raise ValueError(
                "Cannot extract JSON from an empty response."
            )

        start = raw_text.find("{")

        if start == -1:
            raise ValueError(
                "No JSON object was found in the response."
            )

        depth = 0
        inside_string = False
        escaped = False

        for index in range(
            start,
            len(raw_text),
        ):
            character = raw_text[index]

            if inside_string:
                if escaped:
                    escaped = False
                    continue

                if character == "\\":
                    escaped = True
                    continue

                if character == '"':
                    inside_string = False

                continue

            if character == '"':
                inside_string = True
                continue

            if character == "{":
                depth += 1
                continue

            if character == "}":
                depth -= 1

                if depth == 0:
                    return raw_text[
                        start:index + 1
                    ]

        raise ValueError(
            "The response did not contain a complete "
            "balanced JSON object."
        )

    @staticmethod
    def _normalise_groq_payload(
        payload: dict,
    ) -> dict:
        """
        Normalise common response variations returned by Groq.
        """
        # Groq sometimes uses the earlier annotation field name.
        if (
            "annotation_notes" in payload
            and "adjudication_summary" not in payload
        ):
            payload["adjudication_summary"] = (
                payload.pop(
                    "annotation_notes"
                )
            )

        payload.setdefault(
            "adjudication_summary",
            "",
        )

        payload.setdefault(
            "exclusion_reason",
            "",
        )

        payload.setdefault(
            "image_quality",
            "usable",
        )

        label_names = [
            "credential_theft",
            "financial_fraud",
            "malware_distribution",
            (
                "personal_information"
                "_harvesting"
            ),
        ]

        for label in label_names:
            decision = payload.get(
                label,
                {},
            )

            if not isinstance(
                decision,
                dict,
            ):
                decision = {}

            decision.setdefault(
                "present",
                False,
            )

            decision.setdefault(
                "confidence",
                0.0,
            )

            evidence = decision.get(
                "evidence",
                [],
            )

            if isinstance(
                evidence,
                str,
            ):
                evidence = [evidence]

            if not isinstance(
                evidence,
                list,
            ):
                evidence = []

            decision["evidence"] = [
                str(item).strip()
                for item in evidence[:4]
                if str(item).strip()
            ]

            decision.setdefault(
                "decision_reason",
                "",
            )

            payload[label] = decision

        return payload

    @classmethod
    def _parse_groq_output(
        cls,
        raw_text: str,
    ) -> GroqAdjudicationOutput:
        """
        Parse Groq output with recovery for appended malformed text.
        """
        try:
            return (
                GroqAdjudicationOutput
                .model_validate_json(
                    raw_text
                )
            )

        except Exception:
            extracted = (
                cls._extract_balanced_json(
                    raw_text
                )
            )

            payload = json.loads(
                extracted
            )

            payload = (
                cls._normalise_groq_payload(
                    payload
                )
            )

            return (
                GroqAdjudicationOutput
                .model_validate(
                    payload
                )
            )

    @staticmethod
    def _failed_generation_from_error(
        error: BadRequestError,
    ) -> str:
        """
        Retrieve failed_generation from a Groq 400 response.
        """
        body = getattr(
            error,
            "body",
            None,
        )

        if isinstance(body, dict):
            error_body = body.get(
                "error",
                body,
            )

            if isinstance(
                error_body,
                dict,
            ):
                failed_generation = (
                    error_body.get(
                        "failed_generation"
                    )
                )

                if failed_generation:
                    return str(
                        failed_generation
                    )

        response = getattr(
            error,
            "response",
            None,
        )

        if response is not None:
            try:
                response_body = (
                    response.json()
                )

                error_body = (
                    response_body.get(
                        "error",
                        {},
                    )
                )

                failed_generation = (
                    error_body.get(
                        "failed_generation"
                    )
                )

                if failed_generation:
                    return str(
                        failed_generation
                    )

            except Exception:
                pass

        return ""

    @staticmethod
    def _extract_balanced_json(
        raw_text: str,
    ) -> str:
        """
        Extract the first complete, balanced JSON object.

        This removes malformed text that Groq may append after a
        complete JSON object.
        """
        if not raw_text:
            raise ValueError(
                "Cannot extract JSON from an empty response."
            )

        start = raw_text.find("{")

        if start == -1:
            raise ValueError(
                "No JSON object was found in the response."
            )

        depth = 0
        inside_string = False
        escaped = False

        for index in range(
            start,
            len(raw_text),
        ):
            character = raw_text[index]

            if inside_string:
                if escaped:
                    escaped = False
                    continue

                if character == "\\":
                    escaped = True
                    continue

                if character == '"':
                    inside_string = False

                continue

            if character == '"':
                inside_string = True
                continue

            if character == "{":
                depth += 1
                continue

            if character == "}":
                depth -= 1

                if depth == 0:
                    return raw_text[
                        start:index + 1
                    ]

        raise ValueError(
            "The response did not contain a complete "
            "balanced JSON object."
        )

    @staticmethod
    def _normalise_groq_payload(
        payload: dict,
    ) -> dict:
        """
        Normalise common response variations returned by Groq.
        """
        # Groq sometimes uses the earlier annotation field name.
        if (
            "annotation_notes" in payload
            and "adjudication_summary" not in payload
        ):
            payload["adjudication_summary"] = (
                payload.pop(
                    "annotation_notes"
                )
            )

        payload.setdefault(
            "adjudication_summary",
            "",
        )

        payload.setdefault(
            "exclusion_reason",
            "",
        )

        payload.setdefault(
            "image_quality",
            "usable",
        )

        label_names = [
            "credential_theft",
            "financial_fraud",
            "malware_distribution",
            (
                "personal_information"
                "_harvesting"
            ),
        ]

        for label in label_names:
            decision = payload.get(
                label,
                {},
            )

            if not isinstance(
                decision,
                dict,
            ):
                decision = {}

            decision.setdefault(
                "present",
                False,
            )

            decision.setdefault(
                "confidence",
                0.0,
            )

            evidence = decision.get(
                "evidence",
                [],
            )

            if isinstance(
                evidence,
                str,
            ):
                evidence = [evidence]

            if not isinstance(
                evidence,
                list,
            ):
                evidence = []

            decision["evidence"] = [
                str(item).strip()
                for item in evidence[:4]
                if str(item).strip()
            ]

            decision.setdefault(
                "decision_reason",
                "",
            )

            payload[label] = decision

        return payload

    @classmethod
    def _parse_groq_output(
        cls,
        raw_text: str,
    ) -> GroqAdjudicationOutput:
        """
        Parse Groq output with recovery for appended malformed text.
        """
        try:
            return (
                GroqAdjudicationOutput
                .model_validate_json(
                    raw_text
                )
            )

        except Exception:
            extracted = (
                cls._extract_balanced_json(
                    raw_text
                )
            )

            payload = json.loads(
                extracted
            )

            payload = (
                cls._normalise_groq_payload(
                    payload
                )
            )

            return (
                GroqAdjudicationOutput
                .model_validate(
                    payload
                )
            )

    @staticmethod
    def _failed_generation_from_error(
        error: BadRequestError,
    ) -> str:
        """
        Retrieve failed_generation from a Groq 400 response.
        """
        body = getattr(
            error,
            "body",
            None,
        )

        if isinstance(body, dict):
            error_body = body.get(
                "error",
                body,
            )

            if isinstance(
                error_body,
                dict,
            ):
                failed_generation = (
                    error_body.get(
                        "failed_generation"
                    )
                )

                if failed_generation:
                    return str(
                        failed_generation
                    )

        response = getattr(
            error,
            "response",
            None,
        )

        if response is not None:
            try:
                response_body = (
                    response.json()
                )

                error_body = (
                    response_body.get(
                        "error",
                        {},
                    )
                )

                failed_generation = (
                    error_body.get(
                        "failed_generation"
                    )
                )

                if failed_generation:
                    return str(
                        failed_generation
                    )

            except Exception:
                pass

        return ""
    
    def annotate(
        self,
        image_path: str | Path,
        openai_annotation: dict,
        gemini_annotation: dict,
        disagreement_labels: list[str],
    ) -> tuple[
        GroqAdjudicationOutput,
        dict,
    ]:
        image_path = Path(
            image_path
        ).expanduser().resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Adjudication image does not exist: "
                f"{image_path}"
            )

        last_error: Exception | None = None
        last_raw_response = ""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            prompt = self._build_prompt(
                openai_annotation=(
                    openai_annotation
                ),
                gemini_annotation=(
                    gemini_annotation
                ),
                disagreement_labels=(
                    disagreement_labels
                ),
                retry=attempt > 1,
            )

            try:
                completion = (
                    self._make_request(
                        image_path=image_path,
                        prompt=prompt,
                    )
                )

                if not completion.choices:
                    raise ValueError(
                        "Groq returned no completion choices."
                    )

                content = (
                    completion
                    .choices[0]
                    .message
                    .content
                )

                if not content:
                    raise ValueError(
                        "Groq returned an empty response."
                    )

                last_raw_response = content

                parsed = self._parse_groq_output(
    content
)

                usage = getattr(
                    completion,
                    "usage",
                    None,
                )

                usage_dictionary = (
                    usage.model_dump()
                    if usage is not None
                    and hasattr(
                        usage,
                        "model_dump",
                    )
                    else {}
                )

                metadata = {
                    "provider": "groq",
                    "model": self.model,
                    "attempts": attempt,
                    "usage": usage_dictionary,
                }

                return (
                    parsed,
                    metadata,
                )

            except BadRequestError as error:
                last_error = error

                failed_generation = (
                    self
                    ._failed_generation_from_error(
                        error
                    )
                )

                if failed_generation:
                    last_raw_response = (
                        failed_generation
                    )

                    try:
                        parsed = (
                            self
                            ._parse_groq_output(
                                failed_generation
                            )
                        )

                        metadata = {
                            "provider": "groq",
                            "model": self.model,
                            "attempts": attempt,
                            "recovered_from": (
                                "failed_generation"
                            ),
                            "usage": {},
                        }

                        print(
                            "\nRecovered a valid "
                            "adjudication from Groq's "
                            "failed_generation response."
                            f"\nImage: {image_path}",
                            flush=True,
                        )

                        return (
                            parsed,
                            metadata,
                        )

                    except Exception as recovery_error:
                        last_error = (
                            recovery_error
                        )

                print(
                    "\nGroq returned invalid JSON."
                    f"\nImage: {image_path}"
                    f"\nAttempt: "
                    f"{attempt}/"
                    f"{self.max_retries}"
                    f"\nRecovery error: "
                    f"{last_error}",
                    flush=True,
                )

                if (
                    attempt
                    < self.max_retries
                ):
                    time.sleep(
                        5 * attempt
                    )
                
            except Exception as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(attempt)

        debug_directory = (
            PROJECT_ROOT
            / "outputs"
            / "debug"
            / "groq_adjudication"
        )

        debug_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_path = (
            debug_directory
            / (
                f"{image_path.parent.name}_"
                f"{image_path.stem}_"
                "invalid_response.txt"
            )
        )

        debug_path.write_text(
            last_raw_response,
            encoding="utf-8",
        )

        raise RuntimeError(
            "Groq adjudication failed after "
            f"{self.max_retries} attempts. "
            f"Image: {image_path}. "
            f"Last raw response: {debug_path}. "
            f"Final error: {last_error}"
        ) from last_error