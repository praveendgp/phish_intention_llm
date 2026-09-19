from __future__ import annotations
import os
from pathlib import Path
from openai import OpenAI
from .prompt import ANNOTATION_PROMPT
from .schemas import AnnotationOutput
from .utils import image_data_uri

class OpenAIAnnotator:
    def __init__(self, model: str | None = None, max_image_side: int = 1600):
        self.model = model or os.getenv("OPENAI_ANNOTATION_MODEL", "gpt-4.1-mini-2025-04-14")
        self.max_image_side = max_image_side
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def annotate(self, image_path: str | Path) -> tuple[AnnotationOutput, dict]:
        response = self.client.responses.parse(
            model=self.model,
            instructions="Return a rigorous, independent, screenshot-grounded annotation.",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ANNOTATION_PROMPT},
                    {"type": "input_image", "image_url": image_data_uri(image_path, self.max_image_side), "detail": "high"},
                ],
            }],
            text_format=AnnotationOutput,
            max_output_tokens=1200,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed annotation")
        usage = getattr(response, "usage", None)
        usage_dict = usage.model_dump() if usage and hasattr(usage, "model_dump") else {}
        return parsed, {"provider": "openai", "model": self.model, "response_id": response.id, "usage": usage_dict}
