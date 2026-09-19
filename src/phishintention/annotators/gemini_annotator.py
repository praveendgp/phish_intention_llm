from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from .prompt import ANNOTATION_PROMPT
from .schemas import AnnotationOutput
from .utils import normalised_image_bytes


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

load_dotenv(
    dotenv_path=PROJECT_ROOT / ".env"
)


class GeminiAnnotator:
    def __init__(
        self,
        model: str | None = None,
        max_image_side: int = 1600,
        max_retries: int = 3,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.model = (
            model
            or os.getenv(
                "GEMINI_ANNOTATION_MODEL",
                "gemini-3.8-flash",
            )
        )

        self.max_image_side = max_image_side
        self.max_retries = max_retries

        self.client = genai.Client(
            api_key=api_key
        )

    def _request_annotation(
        self,
        image_bytes: bytes,
        prompt: str,
    ):
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        return self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                image_part,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1800,
                response_mime_type="application/json",
                response_schema=AnnotationOutput,
            ),
        )

    def annotate(
        self,
        image_path: str | Path,
    ) -> tuple[AnnotationOutput, dict]:
        image_bytes = normalised_image_bytes(
            image_path,
            self.max_image_side,
        )

        last_error = None
        raw_response = ""

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
            if attempt == 1:
                prompt = ANNOTATION_PROMPT
            else:
                prompt = f"""
{ANNOTATION_PROMPT}

IMPORTANT CORRECTION:

The preceding response could not be parsed because it was incomplete
or did not conform to the required schema.

Return the complete annotation again.

Requirements:
1. Keep every evidence statement short.
2. Return no more than four evidence statements per label.
3. Complete every string, array and object.
4. Return only the required structured JSON.
5. Do not include Markdown or explanatory text.
""".strip()

            try:
                response = self._request_annotation(
                    image_bytes=image_bytes,
                    prompt=prompt,
                )

                raw_response = (
                    response.text or ""
                )

                parsed = response.parsed

                if parsed is None:
                    if not raw_response.strip():
                        raise ValueError(
                            "Gemini returned an empty response."
                        )

                    parsed = (
                        AnnotationOutput
                        .model_validate_json(
                            raw_response
                        )
                    )

                if not isinstance(
                    parsed,
                    AnnotationOutput,
                ):
                    parsed = (
                        AnnotationOutput
                        .model_validate(parsed)
                    )

                usage_metadata = getattr(
                    response,
                    "usage_metadata",
                    None,
                )

                usage = (
                    usage_metadata.model_dump()
                    if usage_metadata
                    and hasattr(
                        usage_metadata,
                        "model_dump",
                    )
                    else {}
                )

                metadata = {
                    "provider": "gemini",
                    "model": self.model,
                    "attempts": attempt,
                    "usage": usage,
                }

                return parsed, metadata

            except (
                ValidationError,
                ValueError,
                TypeError,
            ) as error:
                last_error = error

                if attempt < self.max_retries:
                    time.sleep(attempt)

        debug_directory = (
            PROJECT_ROOT
            / "outputs"
            / "debug"
            / "gemini_annotations"
        )

        debug_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        sample_name = Path(
            image_path
        ).stem

        debug_path = (
            debug_directory
            / f"{sample_name}_invalid_response.txt"
        )

        debug_path.write_text(
            raw_response,
            encoding="utf-8",
        )

        raise ValueError(
            "Gemini failed to return a complete valid annotation "
            f"after {self.max_retries} attempts. "
            f"The last raw response was saved to {debug_path}. "
            f"Final error: {last_error}"
        ) from last_error