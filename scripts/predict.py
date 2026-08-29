import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )


from phishintention.config import Settings
from phishintention.formatter import human_text_report
from phishintention.llm import MockClient, OllamaClient
from phishintention.pipeline import PhishIntentionPipeline
from phishintention.retrieval import KnowledgeBase


argument_parser = argparse.ArgumentParser(
    description=(
        "Analyse a website screenshot for phishing intentions."
    )
)

argument_parser.add_argument(
    "--image",
    required=True,
    help="Path to the website screenshot.",
)

argument_parser.add_argument(
    "--mode",
    choices=[
        "single",
        "always",
        "gated",
    ],
    default="gated",
)

argument_parser.add_argument(
    "--provider",
    choices=[
        "ollama",
        "mock",
    ],
    default="ollama",
)

argument_parser.add_argument(
    "--output-directory",
    default="outputs",
)

arguments = argument_parser.parse_args()


settings = Settings()

if arguments.provider == "mock":
    llm = MockClient()
else:
    llm = OllamaClient(
        settings.base_url,
        settings.model,
        settings.max_image_side,
    )


pipeline = PhishIntentionPipeline(
    llm=llm,
    kb=KnowledgeBase(
        PROJECT_ROOT / "knowledge"
    ),
    threshold=settings.threshold,
    top_k=settings.top_k,
)


result = pipeline.run(
    image_path=arguments.image,
    mode=arguments.mode,
)


output_directory = Path(
    arguments.output_directory
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)


sample_id = result.sample_id

json_path = (
    output_directory
    / f"{sample_id}_result.json"
)

text_path = (
    output_directory
    / f"{sample_id}_report.txt"
)


json_output = json.dumps(
    result.model_dump(mode="json"),
    indent=2,
    ensure_ascii=False,
)

text_output = human_text_report(
    result=result,
    threshold=settings.threshold,
)


json_path.write_text(
    json_output,
    encoding="utf-8",
)

text_path.write_text(
    text_output,
    encoding="utf-8",
)


print()
print(text_output)

print()
print("=" * 72)
print("RAW JSON OUTPUT")
print("=" * 72)
print(json_output)

print()
print(f"Human-readable report saved to: {text_path}")
print(f"JSON result saved to: {json_path}")