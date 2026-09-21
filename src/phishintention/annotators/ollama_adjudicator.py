from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


LABELS = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]


class OllamaLabelDecision(BaseModel):
    present: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: list[str] = Field(
        default_factory=list,
        max_length=4,
    )

    decision_reason: str = Field(
        default="",
        max_length=500,
    )


class OllamaAdjudicationOutput(BaseModel):
    credential_theft: OllamaLabelDecision
    financial_fraud: OllamaLabelDecision
    malware_distribution: OllamaLabelDecision
    personal_information_harvesting: OllamaLabelDecision

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


ADJUDICATION_PROMPT = """
You are a defensive cybersecurity dataset adjudicator.

Review a static phishing-webpage screenshot and two independent
annotations produced by two independent local Ollama models.

Resolve every disagreement using only evidence visibly present in
the current screenshot.

INTENTION DEFINITIONS

credential_theft:
Visible collection of authentication secrets, including username,
login email, password, PIN, OTP, MFA code, security answer or
recovery code.

financial_fraud:
Visible collection of card, bank, payment, transfer, cryptocurrency,
gift-card or other direct financial information, or a visible request
to transfer money.

malware_distribution:
Visible inducement to download, install, open, execute, enable or run
a suspicious application, attachment, executable, APK, command,
macro, fake update or software component.

personal_information_harvesting:
Visible collection of personal or identity information beyond
authentication, such as phone number, address, date of birth,
passport, government ID, tax ID, employment or health information.

RULES

1. Assign each label independently.
2. Multi-label output is allowed.
3. Use only evidence visible in this screenshot.
4. Do not infer a hidden form or a later webpage.
5. A login, payment, download or registration element alone does not
   establish malicious intent without suspicious context.
6. A logo, warning or Confirm My Identity button alone is insufficient.
7. Set image_quality to unusable only when the screenshot is blank,
   unreadable, incomplete or error-only.
8. Confidence measures confidence in the annotation decision.
9. Use short evidence statements.
10. Return only one valid JSON object.

Provider A annotation:

{annotation_a}

Provider B annotation:

{annotation_b}

Disagreement labels:

{disagreement_labels}
""".strip()


class OllamaAdjudicator:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.model = (
            model
            or os.getenv(
                "OLLAMA_ADJUDICATION_MODEL",
                "llama3.2-vision:11b",
            )
        )

        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.timeout = (
            float(timeout)
            if timeout is not None
            else float(
                os.getenv(
                    "OLLAMA_ADJUDICATION_TIMEOUT",
                    "300",
                )
            )
        )

        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(
                os.getenv(
                    "OLLAMA_ADJUDICATION_MAX_RETRIES",
                    "2",
                )
            )
        )

    @staticmethod
    def _encode_image(
        image_path: str | Path,
    ) -> str:
        path = Path(
            image_path
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        return base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

    @staticmethod
    def _extract_balanced_json(
        raw_text: str,
    ) -> str:
        start = raw_text.find("{")

        if start < 0:
            raise ValueError(
                "No JSON object found in Ollama response."
            )

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(raw_text),
        ):
            character = raw_text[index]

            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False

                continue

            if character == '"':
                in_string = True

            elif character == "{":
                depth += 1

            elif character == "}":
                depth -= 1

                if depth == 0:
                    return raw_text[
                        start:index + 1
                    ]

        raise ValueError(
            "Ollama returned incomplete JSON."
        )

    @staticmethod
    def _normalise_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            "annotation_notes" in payload
            and "adjudication_summary"
            not in payload
        ):
            payload[
                "adjudication_summary"
            ] = payload.pop(
                "annotation_notes"
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

        for label in LABELS:
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
    def _parse_response(
        cls,
        raw_text: str,
    ) -> OllamaAdjudicationOutput:
        try:
            return (
                OllamaAdjudicationOutput
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
                cls._normalise_payload(
                    payload
                )
            )

            return (
                OllamaAdjudicationOutput
                .model_validate(
                    payload
                )
            )

    def annotate(
        self,
        image_path: str | Path,
        annotation_a: dict[str, Any],
        annotation_b: dict[str, Any],
        disagreement_labels: list[str],
    ) -> tuple[
        OllamaAdjudicationOutput,
        dict[str, Any],
    ]:
        prompt = (
            ADJUDICATION_PROMPT.format(
                annotation_a=json.dumps(
                    annotation_a,
                    indent=2,
                    ensure_ascii=False,
                ),
                annotation_b=json.dumps(
                    annotation_b,
                    indent=2,
                    ensure_ascii=False,
                ),
                disagreement_labels=json.dumps(
                    disagreement_labels,
                    ensure_ascii=False,
                ),
            )
        )

        encoded_image = self._encode_image(
            image_path
        )

        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        encoded_image,
                    ],
                }
            ],
            "format": (
                OllamaAdjudicationOutput
                .model_json_schema()
            ),
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1500,
            },
            "keep_alive": "10m",
        }

        last_error: Exception | None = None
        last_raw_response = ""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                response = requests.post(
                    (
                        f"{self.base_url}"
                        "/api/chat"
                    ),
                    json=request_body,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                response_payload = (
                    response.json()
                )

                message = (
                    response_payload.get(
                        "message",
                        {},
                    )
                )

                raw_content = str(
                    message.get(
                        "content",
                        "",
                    )
                )

                if not raw_content.strip():
                    raise ValueError(
                        "Ollama returned an empty response."
                    )

                last_raw_response = raw_content

                parsed = (
                    self._parse_response(
                        raw_content
                    )
                )

                metadata = {
                    "provider": "ollama",
                    "model": self.model,
                    "attempts": attempt,
                    "total_duration": (
                        response_payload.get(
                            "total_duration"
                        )
                    ),
                    "load_duration": (
                        response_payload.get(
                            "load_duration"
                        )
                    ),
                    "prompt_eval_count": (
                        response_payload.get(
                            "prompt_eval_count"
                        )
                    ),
                    "eval_count": (
                        response_payload.get(
                            "eval_count"
                        )
                    ),
                    "eval_duration": (
                        response_payload.get(
                            "eval_duration"
                        )
                    ),
                }

                return parsed, metadata

            except Exception as error:
                last_error = error

                print(
                    "\nLocal Ollama adjudication failed."
                    f"\nImage: {image_path}"
                    f"\nAttempt: "
                    f"{attempt}/{self.max_retries}"
                    f"\nError: {error}",
                    flush=True,
                )

                if attempt < self.max_retries:
                    time.sleep(
                        3 * attempt
                    )

        debug_directory = (
            PROJECT_ROOT
            / "outputs"
            / "debug"
            / "ollama_adjudication"
        )

        debug_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_path = (
            debug_directory
            / (
                f"{Path(image_path).parent.name}_"
                f"{Path(image_path).stem}_"
                "invalid_response.txt"
            )
        )

        debug_path.write_text(
            last_raw_response,
            encoding="utf-8",
        )

        raise RuntimeError(
            "Local Ollama adjudication failed "
            f"after {self.max_retries} attempts. "
            f"Image: {image_path}. "
            f"Debug output: {debug_path}. "
            f"Final error: {last_error}"
        ) from last_error