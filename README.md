# PhishIntentionLLM

A defensive, screenshot-based multi-agent Retrieval-Augmented Generation system for identifying the likely intentions of known phishing webpages.

The project predicts one or more of four labels:

- `credential_theft`
- `financial_fraud`
- `malware_distribution`
- `personal_information_harvesting`

This repository uses the earlier custom Python implementation in `pipeline.py`. It does not use LangChain or LangGraph.

## Responsible-use boundary

This project is intended only for defensive cybersecurity research using static, released research data. It does not crawl live phishing websites, execute archived HTML, collect credentials, or generate phishing content.

## Architecture

```text
Screenshot
   ↓
Vision Analysis Agent
   ↓
Common-threat TF-IDF retrieval
   ↓
Initial multi-label classifier
   ↓
Selected category specialists
   ↓
Category-specific TF-IDF retrieval
   ↓
Validation and evidence synthesis
   ↓
Final labels, confidence and trace
```

Inference modes:

- `single`: single-agent baseline
- `always`: invoke every specialist
- `gated`: invoke likely specialists and use confidence-aware feedback

## Annotation and evaluation design

```text
OpenAI GPT-4.1 Mini
          +
Gemini 3.8 Flash
          ↓
OpenAI/Gemini agreement
          ↓ disagreements
Local Gemma 3 12B through Ollama
          ↓
Mandatory human review
          ↓
final_adjudicated.csv
          ↓
manifest.csv ground truth
          ↓
Evaluate Qwen2.5-VL pipeline
```

`Groq` and `Llama 3.2 Vision` are no longer active dependencies:

- Groq was removed because its hosted tie-breaker reached remote token quotas.
- `llama3.2-vision:11b` was removed because the installed Ollama runner returned `unknown model architecture: 'mllama'`.
- The active local tie-breaker is `gemma3:12b`.
- Historical Groq results should be archived, not relabelled or deleted, because they preserve provenance.

Every machine tie-breaker result must remain `requires_human_review=True`. Only a completed human review may produce final adjudicated ground truth.

---

# 1. Project structure

```text
PhishIntentionLLM/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env
├── src/phishintention/
│   ├── pipeline.py
│   ├── retrieval.py
│   ├── metrics.py
│   └── annotators/
│       ├── openai_annotator.py
│       ├── gemini_annotator.py
│       ├── ollama_adjudicator.py
│       ├── consensus.py
│       └── schemas.py
├── scripts/
│   ├── prepare_data.py
│   ├── annotate_llm.py
│   ├── calculate_annotation_agreement.py
│   ├── retry_failed_ollama.py
│   ├── finalise_annotations.py
│   ├── apply_adjudicated_labels.py
│   ├── predict.py
│   ├── evaluate.py
│   └── run_ablation.py
├── knowledge/
│   ├── common.json
│   └── specialists.json
├── data/
│   ├── raw/
│   ├── processed/manifest.csv
│   └── annotations/
└── outputs/
```

---

# 2. Environment setup

```bash
cd "/Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Validate the environment:

```bash
which python
python -c "import sys; print(sys.executable)"
python -m compileall -q src scripts tests
python -m pytest -q
```

The interpreter path should point to `.venv`.

## Active dependencies

The requirements should include at least:

```text
pandas
numpy
Pillow
pydantic
python-dotenv
requests
scikit-learn
streamlit
pytest
openai
google-genai
```

The `groq` package is no longer required:

```bash
python -m pip uninstall -y groq
```

---

# 3. Environment variables

Create or update `.env`:

```dotenv
# Main evaluated Qwen pipeline
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:72b
CONFIDENCE_THRESHOLD=0.55
TOP_K=3
MAX_IMAGE_SIDE=1280

# OpenAI primary annotator
OPENAI_API_KEY=
OPENAI_ANNOTATION_MODEL=gpt-4.1-mini-2025-04-14

# Gemini primary annotator
GEMINI_API_KEY=
GEMINI_ANNOTATION_MODEL=gemini-3.8-flash
GEMINI_ANNOTATION_MAX_RETRIES=3
GEMINI_ANNOTATION_MAX_OUTPUT_TOKENS=1800

# Local Ollama tie-breaker
OLLAMA_ADJUDICATION_MODEL=gemma3:12b
OLLAMA_ADJUDICATION_TIMEOUT=600
OLLAMA_ADJUDICATION_MAX_RETRIES=2
```

Remove obsolete active settings:

```dotenv
# Remove these if present
GROQ_API_KEY=
GROQ_ADJUDICATION_MODEL=
GROQ_MAX_RETRIES=
GROQ_REQUEST_TIMEOUT_SECONDS=
GROQ_RETRY_DELAY_SECONDS=
```

Do not commit `.env`.

---

# 4. Ollama setup

Install and start Ollama:

```bash
brew install ollama
brew services start ollama
```

Or run:

```bash
ollama serve
```

Pull the local adjudication model:

```bash
ollama pull gemma3:12b
```

Pull the main evaluation model if it is not already installed:

```bash
ollama pull qwen2.5vl:72b
```

Verify:

```bash
ollama --version
ollama list
curl -s http://localhost:11434/api/tags | python -m json.tool
```

## Test Gemma text inference

```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:12b",
    "messages": [
      {
        "role": "user",
        "content": "Reply with OK."
      }
    ],
    "stream": false
  }' \
  | python -m json.tool
```

## Test Gemma vision inference

```bash
IMAGE_PATH="/absolute/path/to/screenshot.jpg"
IMAGE_BASE64=$(base64 < "$IMAGE_PATH" | tr -d '\n')

curl -sS http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma3:12b\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"List only visible input fields, buttons, requested information and requested actions. Do not infer a later page.\",
        \"images\": [\"$IMAGE_BASE64\"]
      }
    ],
    \"stream\": false,
    \"options\": {
      \"temperature\": 0,
      \"num_predict\": 300,
      \"num_ctx\": 4096
    }
  }" \
  | python -m json.tool
```

Monitor loaded models:

```bash
ollama ps
```

To release a large model before local adjudication:

```bash
ollama stop qwen2.5vl:72b
```

---

# 5. Dataset preparation

Expected raw-data layout:

```text
data/raw/
├── phish_iris/
└── putra/
```

Back up the manifest before regenerating it:

```bash
cp data/processed/manifest.csv \
   data/processed/manifest_backup.csv
```

Generate the manifest:

```bash
python scripts/prepare_data.py \
  --phish-iris data/raw/phish_iris \
  --putra data/raw/putra
```

Validate the indexed images:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/processed/manifest.csv").fillna("")
missing = df[~df["image_path"].map(lambda value: Path(str(value)).exists())]

print("Manifest rows:", len(df))
print("Unique images:", df["image_path"].nunique())
print("Missing images:", len(missing))
print("\nBy source:")
print(df.groupby("source")["image_path"].nunique())
PY
```

`manifest.csv` is the dataset index and final ground-truth registry. It is not the RAG knowledge base.

---

# 6. Knowledge base and RAG

The RAG knowledge is stored in:

```text
knowledge/common.json
knowledge/specialists.json
```

- `common.json` contains common phishing patterns, boundary rules, abstention rules and benign counterexamples.
- `specialists.json` contains category-specific knowledge for the four intentions.
- `retrieval.py` performs TF-IDF indexing and cosine-similarity retrieval.
- Extracted screenshot evidence becomes the retrieval query.
- Common entries ground the initial classifier.
- Category-filtered specialist entries ground each selected specialist.

Validate KB counts:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("knowledge").glob("*.json")):
    entries = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.name}: {len(entries)} entries")
PY
```

Do not update the KB using final-test examples after inspecting their errors. Use development or validation examples, freeze the KB, and run final evaluation afterwards.

---

# 7. Independent primary annotation

## Pilot with OpenAI

```bash
python scripts/annotate_llm.py \
  --provider openai \
  --limit 10 \
  --output-dir data/annotations
```

## Pilot with Gemini

```bash
python scripts/annotate_llm.py \
  --provider gemini \
  --limit 10 \
  --output-dir data/annotations
```

Expected outputs:

```text
data/annotations/openai_annotations.jsonl
data/annotations/gemini_annotations.jsonl
```

Run the full annotation after validating the pilot:

```bash
python scripts/annotate_llm.py \
  --provider openai \
  --output-dir data/annotations

python scripts/annotate_llm.py \
  --provider gemini \
  --output-dir data/annotations
```

Do not use `--force` unless deliberately replacing a complete run.

Verify coverage:

```bash
python - <<'PY'
import json
from pathlib import Path


def ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {
        json.loads(line)["sample_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

openai_ids = ids("data/annotations/openai_annotations.jsonl")
gemini_ids = ids("data/annotations/gemini_annotations.jsonl")

print("OpenAI:", len(openai_ids))
print("Gemini:", len(gemini_ids))
print("Shared:", len(openai_ids & gemini_ids))
print("Missing from Gemini:", len(openai_ids - gemini_ids))
print("Missing from OpenAI:", len(gemini_ids - openai_ids))
PY
```

---

# 8. Calculate OpenAI/Gemini agreement

```bash
python scripts/calculate_annotation_agreement.py \
  --openai data/annotations/openai_annotations.jsonl \
  --gemini data/annotations/gemini_annotations.jsonl \
  --output-dir data/annotations \
  --create-image-links
```

Outputs:

```text
data/annotations/agreement.csv
data/annotations/agreement_metrics.json
data/annotations/adjudication_queue.csv
data/annotations/adjudication_images/
```

Inspect:

```bash
cat data/annotations/agreement_metrics.json
open data/annotations/adjudication_images
```

---

# 9. Local Ollama tie-breaking with Gemma 3

The active tie-breaker is:

```text
gemma3:12b
```

Use only the failed or unresolved rows. The script should select source values such as:

```text
groq_failed
groq_failed_previous_run
ollama_failed
```

Historical values named `groq_failed` remain valid provenance indicating how the earlier operation failed. Successful local retries should be written as:

```text
adjudication_source = ollama_gemma3_tiebreaker
requires_human_review = True
```

## Test one failed row

```bash
python scripts/retry_failed_ollama.py \
  --sample-id 1a706ce3445291ee \
  --model gemma3:12b \
  --sleep 0
```

## Retry all remaining failed rows

```bash
python scripts/retry_failed_ollama.py \
  --model gemma3:12b \
  --sleep 0
```

Use separate local provenance files:

```text
data/annotations/ollama_adjudications.jsonl
data/annotations/ollama_adjudication_failures.jsonl
```

The local script should update:

```text
data/annotations/final_agreement.csv
```

## Check unresolved rows

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/annotations/final_agreement.csv").fillna("")

failed_sources = {
    "groq_failed",
    "groq_failed_previous_run",
    "ollama_failed",
}

remaining = df[df["adjudication_source"].astype(str).isin(failed_sources)]

print("Remaining failed rows:", len(remaining))

if not remaining.empty:
    print(
        remaining[
            [
                "sample_id",
                "image_path",
                "disagreement_labels",
                "final_annotation_notes",
            ]
        ].to_string(index=False)
    )
PY
```

---

# 10. Historical Groq artefacts

Groq is no longer an active runtime dependency, but historical artefacts should be preserved:

```bash
mkdir -p data/annotations/archive/groq

mv data/annotations/groq_adjudications.jsonl \
   data/annotations/archive/groq/ \
   2>/dev/null || true

mv data/annotations/groq_adjudication_failures.jsonl \
   data/annotations/archive/groq/ \
   2>/dev/null || true
```

Do not globally replace old source labels such as `groq_tiebreaker`. They accurately record how those provisional results were produced.

Archive Groq code rather than deleting it immediately:

```bash
mkdir -p archive/groq

mv src/phishintention/annotators/groq_adjudicator.py \
   archive/groq/ \
   2>/dev/null || true

mv scripts/groq_adjudicate.py \
   archive/groq/ \
   2>/dev/null || true

mv scripts/retry_failed_groq.py \
   archive/groq/ \
   2>/dev/null || true
```

Remove Groq exports from `src/phishintention/annotators/__init__.py` and retain:

```python
from .ollama_adjudicator import (
    OllamaAdjudicationOutput,
    OllamaAdjudicator,
)
```

---

# 11. Generate the human-review queue

Back up an existing queue:

```bash
cp data/annotations/human_review_queue.csv \
   data/annotations/human_review_queue_backup.csv \
   2>/dev/null || true
```

Generate a fresh queue from the latest final agreement:

```bash
python - <<'PY'
import pandas as pd

source = "data/annotations/final_agreement.csv"
target = "data/annotations/human_review_queue.csv"

df = pd.read_csv(source).fillna("")
review_mask = (
    df["requires_human_review"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes", "y"])
)

review = df[review_mask].copy()
review["human_review_completed"] = ""
review["human_review_notes"] = ""
review.to_csv(target, index=False)

print("Human-review rows:", len(review))
print("Created:", target)
PY
```

Create screenshot links:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

queue = pd.read_csv("data/annotations/human_review_queue.csv").fillna("")
output = Path("data/annotations/human_review_images")
output.mkdir(parents=True, exist_ok=True)

created = 0
for position, row in queue.iterrows():
    source = Path(str(row["image_path"])).expanduser()
    if not source.exists():
        print("Missing:", source)
        continue

    target = output / f"{position + 1:04d}_{row['sample_id']}_{source.name}"
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())
    created += 1

print("Created links:", created)
PY
```

Open review files:

```bash
open data/annotations/human_review_queue.csv
open data/annotations/human_review_images
```

For each row, verify or correct:

```text
final_credential_theft
final_financial_fraud
final_malware_distribution
final_personal_information_harvesting
```

Then set:

```text
human_review_completed = 1
human_review_notes = concise evidence-based reason
```

Use only `0` or `1`. Do not infer hidden fields or a later webpage.

---

# 12. Finalise human-approved ground truth

Verify review completeness:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/annotations/human_review_queue.csv").fillna("")
completed = (
    df["human_review_completed"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["1", "true", "yes", "y"])
)

print("Total:", len(df))
print("Completed:", int(completed.sum()))
print("Incomplete:", int((~completed).sum()))
PY
```

Continue only when `Incomplete` is zero:

```bash
python -m py_compile scripts/finalise_annotations.py
python scripts/finalise_annotations.py
```

Output:

```text
data/annotations/final_adjudicated.csv
```

Apply to the manifest:

```bash
cp data/processed/manifest.csv \
   data/processed/manifest_before_adjudication.csv

python scripts/apply_adjudicated_labels.py \
  --manifest data/processed/manifest.csv \
  --adjudicated data/annotations/final_adjudicated.csv
```

---

# 13. Evaluation

The custom `evaluate.py` supports:

```text
--manifest
--mode
--limit
```

It does not accept `--provider` or `--force`.

Check:

```bash
python scripts/evaluate.py --help
```

Run a one-record check:

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  --limit 1
```

Run a five-record check:

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  --limit 5
```

Run full experiments:

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode single

python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode always

python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated
```

Before a fresh run, archive stale outputs because the evaluator has no `--force` option:

```bash
mkdir -p outputs/archive_before_final_evaluation

find outputs \
  -type f \
  \( -name "predictions_*.jsonl" \
  -o -name "metrics_*.json" \
  -o -name "summary_*.csv" \
  -o -name "per_label_*.csv" \) \
  -exec mv {} outputs/archive_before_final_evaluation/ \;
```

Metrics include:

- micro precision
- micro recall
- micro F1
- per-intention precision, recall and F1
- subset accuracy
- Accuracy by Complexity

A `null` complexity score means that the evaluated subset contains no record with that number of positive labels.

---

# 14. Troubleshooting

## `unknown model architecture: 'mllama'`

Do not use `llama3.2-vision:11b` with the affected installed Ollama runner. Use:

```bash
ollama pull gemma3:12b
```

Set:

```dotenv
OLLAMA_ADJUDICATION_MODEL=gemma3:12b
```

## Ollama HTTP 500

Inspect macOS logs:

```bash
tail -n 150 ~/.ollama/logs/server.log

grep -Ei \
  "error|failed|panic|memory|alloc|runner|500|vision" \
  ~/.ollama/logs/server.log \
  | tail -n 100
```

Test text-only and then vision directly before running a batch.

## `ModuleNotFoundError: phishintention`

```bash
python -m pip install -e .
```

## HTML accidentally embedded in a CSV

Back up and clean the CSV before completing human review. Do not treat HTML tags as annotation values.

## All-zero evaluation metrics

Inspect the selected ground-truth row. A zero-positive ground-truth vector can produce `null` complexity groups and zero overlap metrics.

---

# 15. Final reproducibility checklist

- [ ] `.venv` active
- [ ] Dependencies installed
- [ ] Project installed in editable mode
- [ ] Ollama running
- [ ] `gemma3:12b` installed
- [ ] Main Qwen model installed
- [ ] Manifest generated and image paths valid
- [ ] Expanded KB validated
- [ ] OpenAI annotations complete
- [ ] Gemini annotations complete
- [ ] Agreement metrics generated
- [ ] Failed rows retried with local Gemma
- [ ] Historical Groq provenance archived
- [ ] Human-review queue completed
- [ ] `final_adjudicated.csv` created
- [ ] Final labels applied to `manifest.csv`
- [ ] Stale evaluation outputs archived
- [ ] Single, always and gated runs completed
- [ ] Annotation and evaluation artefacts frozen

## Recommended methodology wording

> GPT-4.1 Mini and Gemini 3.8 Flash produced independent primary screenshot annotations. Disagreement or unresolved records were provisionally reviewed using Gemma 3 12B through local Ollama. Every local tie-breaker output remained flagged for human verification. Only final human-approved labels were applied to the manifest and used as ground truth for evaluating the Qwen2.5-VL multi-agent RAG architecture. Historical Groq outputs were retained only for provenance and were not treated as human adjudication.
