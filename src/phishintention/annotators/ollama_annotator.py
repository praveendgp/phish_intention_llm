from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


class LabelDecision(BaseModel):
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


class LocalAnnotationOutput(BaseModel):
    credential_theft: LabelDecision
    financial_fraud: LabelDecision
    malware_distribution: LabelDecision
    personal_information_harvesting: LabelDecision

    image_quality: str = Field(
        pattern="^(usable|unusable)$"
    )

    exclusion_reason: str = Field(
        default="",
        max_length=500,
    )

    annotation_notes: str = Field(
        default="",
        max_length=1000,
    )


ANNOTATION_PROMPT = """
You are annotating static screenshots from a defensive phishing
research dataset.

Assign every intention independently.

INTENTION DEFINITIONS

credential_theft:
Visible collection of authentication secrets, including username,
login email, password, PIN, passcode, OTP, MFA code, security
answer or recovery code.

financial_fraud:
Visible collection of card, bank, payment, transfer,
cryptocurrency, gift-card or other direct financial data, or a
visible request to transfer money.

malware_distribution:
Visible inducement to download, install, open, execute, enable or
run a suspicious application, attachment, executable, APK, command,
macro, fake update or software component.

personal_information_harvesting:
Visible collection of personal or identity information beyond
authentication, including phone number, address, date of birth,
passport, government ID, tax ID, employment or health information.

RULES

1. Evaluate all four intentions.
2. Multiple intentions may be present.
3. Use only evidence visible in the current screenshot.
4. Do not infer forms or behaviour on a later page.
5. A logo, warning or generic button alone is insufficient.
6. A normal login form alone does not prove credential theft unless
   the input is already established as phishing or suspicious context
   is visible.
7. A normal checkout form alone does not prove financial fraud.
8. A normal download button alone does not prove malware distribution.
9. A normal registration or delivery form alone does not prove
   personal-information harvesting.
10. Confidence represents confidence in the annotation decision.
11. Keep evidence concise.
12. Return exactly one JSON object matching the requested schema.
""".strip()


class OllamaAnnotator:
    def __init__(
        self,
        model: str,
        provider_name: str,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_image_side: int | None = None,
    ):
        self.model = model

        self.provider_name = provider_name

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
                    "OLLAMA_ANNOTATION_TIMEOUT",
                    "900",
                )
            )
        )

        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(
                os.getenv(
                    "OLLAMA_ANNOTATION_MAX_RETRIES",
                    "2",
                )
            )
        )

        self.max_image_side = (
            int(max_image_side)
            if max_image_side is not None
            else int(
                os.getenv(
                    "MAX_IMAGE_SIDE",
                    "1280",
                )
            )
        )

    def _encode_image(
        self,
        image_path: str | Path,
    ) -> str:
        path = Path(
            image_path
        ).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        with Image.open(path) as image:
            image = image.convert("RGB")

            image.thumbnail(
                (
                    self.max_image_side,
                    self.max_image_side,
                )
            )

            buffer = BytesIO()

            image.save(
                buffer,
                format="JPEG",
                quality=85,
                optimize=True,
            )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    @staticmethod
    def _extract_balanced_json(
        raw_text: str,
    ) -> str:
        start = raw_text.find("{")

        if start < 0:
            raise ValueError(
                "No JSON object found."
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

                elif character == "\\":
                    escaped = True

                elif character == '"':
                    inside_string = False

                continue

            if character == '"':
                inside_string = True

            elif character == "{":
                depth += 1

            elif character == "}":
                depth -= 1

                if depth == 0:
                    return raw_text[
                        start:index + 1
                    ]

        raise ValueError(
            "Incomplete JSON object."
        )

    @staticmethod
    def _normalise_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        labels = [
            "credential_theft",
            "financial_fraud",
            "malware_distribution",
            "personal_information_harvesting",
        ]

        for label in labels:
            decision = payload.get(
                label,
                {},
            )

            if not isinstance(decision, dict):
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

            if isinstance(evidence, str):
                evidence = [evidence]

            if not isinstance(evidence, list):
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

        payload.setdefault(
            "image_quality",
            "usable",
        )

        payload.setdefault(
            "exclusion_reason",
            "",
        )

        payload.setdefault(
            "annotation_notes",
            "",
        )

        return payload

    @classmethod
    def _parse_response(
        cls,
        raw_text: str,
    ) -> LocalAnnotationOutput:
        try:
            return (
                LocalAnnotationOutput
                .model_validate_json(
                    raw_text
                )
            )

        except Exception:
            json_object = (
                cls._extract_balanced_json(
                    raw_text
                )
            )

            payload = json.loads(
                json_object
            )

            payload = cls._normalise_payload(
                payload
            )

            return (
                LocalAnnotationOutput
                .model_validate(
                    payload
                )
            )

    def annotate(
        self,
        image_path: str | Path,
    ) -> tuple[
        LocalAnnotationOutput,
        dict[str, Any],
    ]:
        encoded_image = self._encode_image(
            image_path
        )

        request_body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": ANNOTATION_PROMPT,
                    "images": [
                        encoded_image,
                    ],
                }
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 1200,
                "num_ctx": 4096,
            },
            "keep_alive": "5m",
        }

        last_error: Exception | None = None
        last_raw_response = ""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=request_body,
                    timeout=self.timeout,
                )

                if not response.ok:
                    raise RuntimeError(
                        "Ollama request failed."
                        f"\nStatus: {response.status_code}"
                        f"\nModel: {self.model}"
                        f"\nResponse: {response.text[:4000]}"
                    )

                response_payload = (
                    response.json()
                )

                raw_content = str(
                    response_payload.get(
                        "message",
                        {},
                    ).get(
                        "content",
                        "",
                    )
                )

                if not raw_content.strip():
                    raise ValueError(
                        "Ollama returned an empty response."
                    )

                last_raw_response = (
                    raw_content
                )

                parsed = self._parse_response(
                    raw_content
                )

                metadata = {
                    "provider": (
                        self.provider_name
                    ),
                    "runtime": "ollama",
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
                    "\nLocal annotation failed."
                    f"\nProvider: {self.provider_name}"
                    f"\nModel: {self.model}"
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
            / "local_annotation"
            / self.provider_name
        )

        debug_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        debug_path = (
            debug_directory
            / (
                f"{Path(image_path).stem}"
                "_invalid_response.txt"
            )
        )

        debug_path.write_text(
            last_raw_response,
            encoding="utf-8",
        )

        raise RuntimeError(
            "Local annotation failed after "
            f"{self.max_retries} attempts. "
            f"Model: {self.model}. "
            f"Image: {image_path}. "
            f"Debug output: {debug_path}. "
            f"Final error: {last_error}"
        ) from last_error