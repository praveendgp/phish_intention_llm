import json
from pathlib import Path


def sample_ids(path):
    source = Path(path)

    if not source.exists():
        return set()

    return {
        json.loads(line)["sample_id"]
        for line in source.read_text().splitlines()
        if line.strip()
    }


openai_ids = sample_ids(
    "data/annotations/"
    "openai_annotations.jsonl"
)

gemini_ids = sample_ids(
    "data/annotations/"
    "gemini_annotations.jsonl"
)

print(
    "OpenAI annotations:",
    len(openai_ids),
)

print(
    "Gemini annotations:",
    len(gemini_ids),
)

print(
    "Shared annotations:",
    len(openai_ids & gemini_ids),
)

print(
    "Missing from Gemini:",
    sorted(openai_ids - gemini_ids),
)

print(
    "Missing from OpenAI:",
    sorted(gemini_ids - openai_ids),
)