import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib import response

import requests
from PIL import Image


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        max_side: int = 1600,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_side = max_side

    def _image_b64(
        self,
        path: str | Path,
    ) -> str:
        """
        Resize and encode an image for Ollama.
        """
        image = Image.open(path).convert("RGB")

        image.thumbnail(
            (
                self.max_side,
                self.max_side,
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
    def _remove_code_fences(
        text: str,
    ) -> str:
        """
        Remove Markdown code fences occasionally returned by a model.
        """
        cleaned = text.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        return cleaned.strip()

    @staticmethod
    def _extract_json_object(
        text: str,
    ) -> str | None:
        """
        Extract the first balanced JSON object from model output.

        This is safer than the previous greedy regular expression because
        it understands nested braces and quoted strings.
        """
        start = text.find("{")

        if start == -1:
            return None

        depth = 0
        inside_string = False
        escaped = False

        for index in range(
            start,
            len(text),
        ):
            character = text[index]

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
                    return text[start:index + 1]

        return None

    @staticmethod
    def _parse_json(
        raw: str,
    ) -> dict[str, Any]:
        """
        Parse a JSON response, allowing surrounding explanatory text.
        """
        cleaned = OllamaClient._remove_code_fences(
            raw
        )

        try:
            parsed = json.loads(cleaned)

            if not isinstance(parsed, dict):
                raise ValueError(
                    "The model returned JSON, but the root value "
                    "was not an object."
                )

            return parsed

        except json.JSONDecodeError:
            extracted = OllamaClient._extract_json_object(
                cleaned
            )

            if extracted is None:
                raise ValueError(
                    "No complete JSON object was found in the "
                    "model response."
                )

            parsed = json.loads(extracted)

            if not isinstance(parsed, dict):
                raise ValueError(
                    "The extracted JSON root was not an object."
                )

            return parsed

    def _generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        num_predict: int = 2048,
    ) -> str:
        """
        Send one generation request to Ollama.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": num_predict,
            },
            "keep_alive": "10m",
        }

        if image_path is not None:
            payload["images"] = [
                self._image_b64(image_path)
            ]

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,
        )

        url = f"{self.base_url.rstrip('/')}/api/generate"

        # print("OLLAMA URL:", url)
        # print("OLLAMA MODEL:", self.model)
        # print("IMAGE PATH:", image_path)

        response = requests.post(
            url,
            json=payload,
            timeout=600,
        )

        if not response.ok:
            raise RuntimeError(
                "\nOllama request failed:\n"
                f"URL: {response.url}\n"
                f"Status: {response.status_code}\n"
                f"Model: {self.model}\n"
                f"Image: {image_path}\n"
                f"Response: {response.text}\n"
            )


        response_body = response.json()

        raw_response = response_body.get(
            "response",
            "",
        )

        if not isinstance(raw_response, str):
            raise ValueError(
                "Ollama returned a non-text response field."
            )

        if not raw_response.strip():
            raise ValueError(
                "Ollama returned an empty response."
            )

        return raw_response

    def ask(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Request structured JSON from Ollama.

        If the first response is incomplete or malformed, perform one
        corrective retry using the original image and stricter instructions.
        """
        structured_prompt = f"""
{prompt}

STRICT OUTPUT REQUIREMENTS:

1. Return exactly one complete JSON object.
2. Do not return Markdown.
3. Do not use JSON code fences.
4. Do not include text before or after the JSON object.
5. Use valid double-quoted JSON keys and string values.
6. Close every array, string, and object.
7. Keep evidence concise.
8. Do not repeat visible text unnecessarily.
9. Limit each evidence list to at most 10 short items.
10. The complete response must be valid for Python json.loads().
""".strip()

        first_raw = self._generate(
            prompt=structured_prompt,
            image_path=image_path,
            num_predict=2048,
        )

        try:
            return self._parse_json(
                first_raw
            )

        except (
            json.JSONDecodeError,
            ValueError,
        ) as first_error:
            repair_prompt = f"""
{structured_prompt}

Your previous response could not be parsed because it was incomplete
or invalid.

Return the analysis again as one smaller, complete JSON object.

Important:
- Shorten visible_text to the most security-relevant visible text.
- Use no more than 8 UI elements.
- Use no more than 6 visual cues.
- Use no more than 4 brand signals.
- Each array item must be a short plain string.
- Return JSON only.
- Ensure all braces and arrays are closed.

Previous invalid response, provided only to help correct formatting:

{first_raw[:4000]}
""".strip()

            second_raw = self._generate(
                prompt=repair_prompt,
                image_path=image_path,
                num_predict=4096,
            )

            try:
                return self._parse_json(
                    second_raw
                )

            except (
                json.JSONDecodeError,
                ValueError,
            ) as second_error:
                diagnostic_directory = Path(
                    "outputs/debug"
                )

                diagnostic_directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    diagnostic_directory
                    / "last_invalid_ollama_response.txt"
                ).write_text(
                    second_raw,
                    encoding="utf-8",
                )

                raise ValueError(
                    "Ollama did not return a complete valid JSON "
                    "object after two attempts. The final raw response "
                    "was saved to "
                    "'outputs/debug/"
                    "last_invalid_ollama_response.txt'. "
                    f"First parsing error: {first_error}. "
                    f"Second parsing error: {second_error}."
                ) from second_error


class MockClient:
    """
    Deterministic model client used for tests and UI demonstration.

    The matching order is important. Validator prompts may include
    specialist reports, so validator detection must happen before
    specialist detection.
    """

    def ask(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> dict[str, Any]:
        prompt_lower = prompt.lower()

        # Layer 1: Vision Analysis Agent
        if "website-screenshot vision analyst" in prompt_lower:
            return {
                "visible_text": (
                    "Demo Sign In Email Password Login"
                ),
                "ui_elements": [
                    "Email input field",
                    "Password input field",
                    "Login button",
                ],
                "visual_cues": [
                    "Sign-in form",
                ],
                "brand_signals": [
                    "Demo website",
                ],
            }

        # Layer 3: Initial Classification Agent
        if (
            "initial multi-label phishing-intention classifier"
            in prompt_lower
        ):
            return {
                "candidates": [
                    {
                        "intent": "credential_theft",
                        "confidence": 0.86,
                        "evidence": [
                            "A password input field is visible.",
                            "The webpage contains a sign-in form.",
                        ],
                    }
                ]
            }

        # Layer 5: Validation and Synthesis Agent
        #
        # IMPORTANT:
        # This condition must remain above the specialist condition.
        # The validator prompt contains the phrase "specialist reports".
        if (
            "validation and synthesis agent"
            in prompt_lower
            or "you are the validator"
            in prompt_lower
        ):
            return {
                "labels": [
                    "credential_theft",
                ],
                "confidence": 0.87,
                "evidence": {
                    "credential_theft": [
                        "The webpage contains an email input field.",
                        "The webpage contains a password input field.",
                        "A login button is visible.",
                    ]
                },
                "evidence_consistency": 1.0,
            }

        # Layer 4: Specialist Agent
        if (
            "defensive specialist" in prompt_lower
            or (
                "specialist" in prompt_lower
                and "specialist reports" not in prompt_lower
            )
        ):
            return {
                "supported": True,
                "confidence": 0.88,
                "evidence": [
                    (
                        "The login form contains a "
                        "password input field."
                    )
                ],
            }

        # Single-agent baseline
        if "single defensive analyst" in prompt_lower:
            return {
                "labels": [
                    "credential_theft",
                ],
                "confidence": 0.80,
                "evidence": {
                    "credential_theft": [
                        "A password input field is visible."
                    ]
                },
                "evidence_consistency": 1.0,
            }

        raise ValueError(
            "MockClient received an unrecognised prompt. "
            f"Prompt beginning: {prompt[:200]!r}"
        )