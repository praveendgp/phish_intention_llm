# PhishIntentionLLM

A defensive, screenshot-based multi-agent Retrieval-Augmented Generation (RAG) system for identifying the likely intentions of known phishing webpages.

The project analyses static webpage screenshots and predicts one or more of four phishing-intention labels:

- `credential_theft`
- `financial_fraud`
- `malware_distribution`
- `personal_information_harvesting`

The project uses the earlier custom Python orchestration implemented in `pipeline.py`. It does **not** use LangChain or LangGraph.

---

## 1. Scope and responsible-use boundary

This repository is intended only for defensive cybersecurity research using released, static research datasets.

The project:

- analyses static screenshots;
- does not crawl live phishing websites;
- does not execute archived HTML;
- does not collect credentials;
- does not generate phishing content;
- does not independently prove that an arbitrary website is phishing.

The four-intention classifier should normally receive screenshots already known to be phishing. Legitimate pages should not be included in the final four-intention evaluation.

---

## 2. System architecture

```text
Website screenshot
        ↓
Vision Analysis Agent
        ↓
Screenshot-visible text, UI elements, visual cues and brand signals
        ↓
Context Enrichment using TF-IDF RAG
        ↓
Initial Multi-label Classification Agent
        ↓
Confidence-gated Specialist Agents
        ├── Credential Theft Specialist
        ├── Financial Fraud Specialist
        ├── Malware Distribution Specialist
        └── Personal Information Harvesting Specialist
        ↓
Validation and Synthesis Agent
        ↓
Optional low-confidence feedback pass
        ↓
Final labels, confidence, evidence and execution trace
```

Three inference modes are available:

- `single`: single-agent baseline;
- `always`: all four specialist agents are invoked;
- `gated`: only sufficiently likely specialists are invoked, with a low-confidence feedback pass where required.

---

## 3. Main technologies

- Python 3.11
- Ollama
- Qwen2.5-VL through Ollama
- TF-IDF and cosine similarity for RAG
- Pydantic for structured output validation
- Streamlit for analysis and annotation
- scikit-learn for evaluation metrics
- OpenAI API for independent annotation
- Gemini API for independent annotation
- Groq API as a machine tie-breaker for model disagreements

> Groq-resolved rows must still be marked for human review. Groq is not a substitute for final human adjudication.

---

## 4. Repository structure

```text
PhishIntentionLLM/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env
├── .env.example
│
├── src/phishintention/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── formatter.py
│   ├── llm.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── retrieval.py
│   ├── schema.py
│   └── annotators/
│       ├── __init__.py
│       ├── schemas.py
│       ├── prompt.py
│       ├── utils.py
│       ├── openai_annotator.py
│       ├── gemini_annotator.py
│       ├── groq_adjudicator.py
│       └── consensus.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── annotate_llm.py
│   ├── calculate_annotation_agreement.py
│   ├── groq_adjudicate.py
│   ├── finalise_annotations.py
│   └── apply_adjudicated_labels.py
│
├── knowledge/
│   ├── common.json
│   └── specialists.json
│
├── data/
│   ├── raw/
│   │   ├── phish_iris/
│   │   └── putra/
│   ├── processed/
│   │   └── manifest.csv
│   └── annotations/
│       ├── openai_annotations.jsonl
│       ├── gemini_annotations.jsonl
│       ├── agreement.csv
│       ├── agreement_metrics.json
│       ├── groq_adjudications.jsonl
│       ├── groq_adjudication_failures.jsonl
│       ├── final_agreement.csv
│       ├── human_review_queue.csv
│       └── final_adjudicated.csv
│
├── outputs/
│   ├── predictions/
│   ├── reports/
│   ├── evaluation/
│   └── debug/
│
└── tests/
```

---

## 5. Environment setup

### 5.1 Enter the project

```bash
cd "/Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM"
```

### 5.2 Create and activate the virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If both Conda base and `.venv` are active, the prompt may show `(.venv) (base)`. Confirm that the selected Python belongs to `.venv`:

```bash
which python
python -c "import sys; print(sys.executable)"
```

### 5.3 Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

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
groq
```

### 5.4 Validate imports and source code

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

---

## 6. Environment variables

Create `.env` in the project root.

```dotenv
# Main Qwen/Ollama evaluation pipeline
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl:72b
CONFIDENCE_THRESHOLD=0.55
TOP_K=3
MAX_IMAGE_SIDE=1280

# OpenAI annotator
OPENAI_API_KEY=
OPENAI_ANNOTATION_MODEL=gpt-4.1-mini-2025-04-14

# Gemini annotator
GEMINI_API_KEY=
GEMINI_ANNOTATION_MODEL=gemini-3.8-flash
GEMINI_ANNOTATION_MAX_RETRIES=3
GEMINI_ANNOTATION_MAX_OUTPUT_TOKENS=1800

# Groq tie-breaker
GROQ_API_KEY=
GROQ_ADJUDICATION_MODEL=qwen/qwen3.8-27b
GROQ_MAX_RETRIES=3
GROQ_REQUEST_TIMEOUT_SECONDS=90
GROQ_RETRY_DELAY_SECONDS=20
```

Do not commit `.env`.

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
outputs/
data/raw/
```

---

## 7. Ollama and Qwen setup

Install and start Ollama.

```bash
brew install ollama
brew services start ollama
```

Alternatively:

```bash
ollama serve
```

Pull the configured Qwen model:

```bash
ollama pull qwen2.5vl:72b
```

Verify the model and service:

```bash
ollama list
curl http://localhost:11434/api/tags
```

Check the project configuration:

```bash
grep -E 'OLLAMA_MODEL|OLLAMA_BASE_URL|CONFIDENCE_THRESHOLD|TOP_K' .env
```

---

## 8. Dataset layout

### 8.1 Phish-IRIS

```text
data/raw/phish_iris/
└── phishIRIS_DL_Dataset/
    ├── train/
    │   ├── amazon/
    │   ├── paypal/
    │   ├── other/
    │   └── ...
    └── val/
        ├── amazon/
        ├── paypal/
        ├── other/
        └── ...
```

The Phish-IRIS folders are brand classes. `phish_iris` is a dataset-source value, not an intention label.

### 8.2 Putra

```text
data/raw/putra/
├── phishing_0001-0500/
│   └── <website_id>/
│       ├── screenshots/
│       │   ├── original_js_on.jpg
│       │   ├── original_js_off.jpg
│       │   ├── index_js_on.jpg
│       │   ├── index_js_off.jpg
│       │   ├── clean_js_on.jpg
│       │   └── clean_js_off.jpg
│       ├── original.html
│       ├── index.html
│       ├── clean.html
│       └── asset_details.json
└── not-phishing_0001-0500/
    └── <website_id>/
        └── ...
```

The project normally selects one canonical screenshot per Putra website, preferring `original_js_on.jpg`.

---

## 9. Generate `manifest.csv`

Back up an existing annotated manifest before regenerating it:

```bash
cp data/processed/manifest.csv \
   data/processed/manifest_backup.csv
```

Generate or regenerate the manifest:

```bash
python scripts/prepare_data.py \
  --phish-iris data/raw/phish_iris \
  --putra data/raw/putra
```

Expected file:

```text
data/processed/manifest.csv
```

Verify paths and source counts:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("data/processed/manifest.csv")
df = pd.read_csv(path).fillna("")

print("Total records:", len(df))
print("\nSources:")
print(df["source"].value_counts(dropna=False))
print("\nPhishing status:")
print(df["phishing_status"].value_counts(dropna=False))

missing = df[
    ~df["image_path"].map(
        lambda value: Path(str(value)).exists()
    )
]

print("\nMissing image paths:", len(missing))
PY
```

Required result:

```text
Missing image paths: 0
```

---

## 10. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application supports:

- screenshot analysis;
- human-readable output;
- JSON output;
- manual ground-truth annotation;
- evaluation-result viewing where implemented.

---

## 11. Test the custom pipeline

### 11.1 Run unit tests

```bash
python -m pytest -q
```

### 11.2 Mock inference

```bash
python scripts/predict.py \
  --image tests/fixtures/sample_login.png \
  --provider mock \
  --mode gated
```

### 11.3 Real Qwen inference

```bash
python scripts/predict.py \
  --image "path/to/original_js_on.jpg" \
  --provider ollama \
  --mode gated
```

---

# Annotation and ground-truth workflow

## 12. Annotation methodology

The final evaluation ground truth is created as follows:

```text
Known phishing screenshots
        ↓
OpenAI independent annotation
        +
Gemini independent annotation
        ↓
Agreement calculation
        ↓
Direct agreement or disagreement
        ↓
Groq machine tie-breaker for disagreements
        ↓
Human review of every Groq-resolved or failed row
        ↓
final_adjudicated.csv
        ↓
Apply labels to manifest.csv
```

The evaluated Qwen pipeline must not be shown to OpenAI or Gemini annotators. Do not use Qwen predictions as ground truth.

---

## 13. Run a small annotation pilot

Use the same manifest subset for both providers.

### 13.1 OpenAI pilot

```bash
python scripts/annotate_llm.py \
  --provider openai \
  --limit 10 \
  --output-dir data/annotations
```

### 13.2 Gemini pilot

```bash
python scripts/annotate_llm.py \
  --provider gemini \
  --limit 10 \
  --output-dir data/annotations
```

Expected files:

```text
data/annotations/openai_annotations.jsonl
data/annotations/gemini_annotations.jsonl
```

Failures are written to:

```text
data/annotations/openai_failures.jsonl
data/annotations/gemini_failures.jsonl
```

The annotation scripts resume by skipping sample IDs already present in the provider JSONL file.

---

## 14. Verify provider coverage

```bash
python - <<'PY'
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
    "data/annotations/openai_annotations.jsonl"
)

gemini_ids = sample_ids(
    "data/annotations/gemini_annotations.jsonl"
)

print("OpenAI annotations:", len(openai_ids))
print("Gemini annotations:", len(gemini_ids))
print("Shared annotations:", len(openai_ids & gemini_ids))
print("Missing from Gemini:", sorted(openai_ids - gemini_ids))
print("Missing from OpenAI:", sorted(gemini_ids - openai_ids))
PY
```

For a 10-record pilot, the desired result is:

```text
OpenAI annotations: 10
Gemini annotations: 10
Shared annotations: 10
Missing from Gemini: []
Missing from OpenAI: []
```

---

## 15. Run the full annotation

After validating the pilot:

```bash
python scripts/annotate_llm.py \
  --provider openai \
  --output-dir data/annotations
```

```bash
python scripts/annotate_llm.py \
  --provider gemini \
  --output-dir data/annotations
```

Do not use `--force` unless intentionally replacing an annotation run.

---

## 16. Calculate OpenAI/Gemini agreement

```bash
python scripts/calculate_annotation_agreement.py \
  --openai data/annotations/openai_annotations.jsonl \
  --gemini data/annotations/gemini_annotations.jsonl \
  --output-dir data/annotations \
  --create-image-links
```

Expected outputs:

```text
data/annotations/agreement.csv
data/annotations/agreement_metrics.json
data/annotations/adjudication_queue.csv
data/annotations/adjudication_images/
```

Inspect the agreement summary:

```bash
cat data/annotations/agreement_metrics.json
```

Open the disagreement-image folder:

```bash
open data/annotations/adjudication_images
```

---

## 17. Use Groq as a machine tie-breaker

Groq is used only when OpenAI and Gemini disagree or disagree about image quality.

### 17.1 Run the pilot tie-breaker

```bash
python scripts/groq_adjudicate.py \
  --limit 10 \
  --sleep 20
```

### 17.2 Resume and retry previous failures

```bash
python scripts/groq_adjudicate.py \
  --limit 10 \
  --sleep 20 \
  --retry-failed
```

### 17.3 Run the full tie-breaker

```bash
python scripts/groq_adjudicate.py \
  --sleep 20
```

Do not use `--force` when resuming. Successful decisions are cached in:

```text
data/annotations/groq_adjudications.jsonl
```

Failures are stored in:

```text
data/annotations/groq_adjudication_failures.jsonl
```

Final combined output:

```text
data/annotations/final_agreement.csv
```

Decision-source values may include:

```text
openai_gemini_agreement
groq_tiebreaker
groq_tiebreaker_cached
groq_failed
groq_failed_previous_run
```

Every Groq-resolved or Groq-failed row must retain:

```text
requires_human_review = True
```

---

## 18. Inspect `final_agreement.csv`

```bash
python - <<'PY'
import pandas as pd

path = "data/annotations/final_agreement.csv"
df = pd.read_csv(path).fillna("")

print("Total records:", len(df))
print("\nDecision sources:")
print(df["adjudication_source"].value_counts(dropna=False))

review = (
    df["requires_human_review"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes"])
)

print("\nRecords requiring human review:", int(review.sum()))

for label in [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]:
    column = f"final_{label}"
    missing = df[column].astype(str).str.strip().eq("").sum()
    print(f"{column} missing: {missing}")
PY
```

---

## 19. Create the human-review queue

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
    .isin(["true", "1", "yes"])
)

review = df[review_mask].copy()
review["human_review_completed"] = ""
review["human_review_notes"] = ""
review.to_csv(target, index=False)

print("Review records:", len(review))
print("Created:", target)
PY
```

---

## 20. Create links to human-review images

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

queue_path = Path("data/annotations/human_review_queue.csv")
output = Path("data/annotations/human_review_images")
output.mkdir(parents=True, exist_ok=True)

queue = pd.read_csv(queue_path).fillna("")
created = 0

for position, row in queue.iterrows():
    source = Path(str(row["image_path"])).expanduser()
    if not source.exists():
        print("Missing:", source)
        continue

    target = output / (
        f"{position + 1:04d}_"
        f"{row['sample_id']}_"
        f"{source.name}"
    )

    if target.exists() or target.is_symlink():
        target.unlink()

    target.symlink_to(source.resolve())
    created += 1

print(f"Created {created} links under {output}")
PY
```

Open the folder:

```bash
open data/annotations/human_review_images
```

---

## 21. Complete human review

Open:

```bash
open data/annotations/human_review_queue.csv
```

For each row:

1. Open the screenshot from `image_path`.
2. Review the provider evidence and Groq decision.
3. Correct these columns where required:

```text
final_credential_theft
final_financial_fraud
final_malware_distribution
final_personal_information_harvesting
```

4. Fill:

```text
human_review_completed = 1
human_review_notes = concise evidence-based explanation
```

Use only:

```text
1 = present
0 = absent
```

Do not treat a generic lure such as “Confirm My Identity” as sufficient evidence of Credential Theft unless the visible screenshot supports collection of authentication secrets.

---

## 22. Verify completion of human review

```bash
python - <<'PY'
import pandas as pd

path = "data/annotations/human_review_queue.csv"
df = pd.read_csv(path).fillna("")

completed = (
    df["human_review_completed"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["1", "true", "yes", "y"])
)

print("Total reviews:", len(df))
print("Completed:", int(completed.sum()))
print("Incomplete:", int((~completed).sum()))

if not completed.all():
    print("\nIncomplete records:")
    print(
        df.loc[
            ~completed,
            [
                "image_path",
                "adjudication_source",
                "disagreement_labels",
            ],
        ].to_string(index=False)
    )
PY
```

Do not continue until:

```text
Incomplete: 0
```

---

## 23. Create final adjudicated ground truth

```bash
python -m py_compile scripts/finalise_annotations.py
python scripts/finalise_annotations.py
```

Expected output:

```text
data/annotations/final_adjudicated.csv
```

If finalisation stops, inspect:

```text
data/annotations/incomplete_adjudication.csv
```

---

## 24. Apply adjudicated labels to `manifest.csv`

Back up the manifest:

```bash
cp data/processed/manifest.csv \
   data/processed/manifest_before_adjudication.csv
```

Apply labels:

```bash
python scripts/apply_adjudicated_labels.py \
  --manifest data/processed/manifest.csv \
  --adjudicated data/annotations/final_adjudicated.csv
```

Verify:

```bash
python - <<'PY'
import pandas as pd

labels = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

df = pd.read_csv("data/processed/manifest.csv").fillna("")

adjudicated = df[
    df["annotation_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("adjudicated")
].copy()

print("Adjudicated records:", len(adjudicated))

for label in labels:
    values = pd.to_numeric(
        adjudicated[label],
        errors="coerce",
    ).fillna(0)
    print(f"{label}: {int(values.sum())}")
PY
```

---

# Evaluation workflow

## 25. Understand the current evaluator interface

The earlier custom evaluator supports:

```text
--manifest
--mode
--limit
```

Check it with:

```bash
python scripts/evaluate.py --help
```

Do not pass unsupported options such as:

```text
--provider
--force
```

The evaluator reads Ollama settings from `.env`.

---

## 26. Check the evaluation ground truth

```bash
python - <<'PY'
import pandas as pd

labels = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

df = pd.read_csv("data/processed/manifest.csv").fillna("")

eligible = df[
    df["annotation_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("adjudicated")
    & df["phishing_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("phishing")
].copy()

for label in labels:
    eligible[label] = pd.to_numeric(
        eligible[label],
        errors="coerce",
    ).fillna(0).astype(int)

eligible["label_count"] = eligible[labels].sum(axis=1)

print("Eligible records:", len(eligible))
print("\nLabel-count distribution:")
print(eligible["label_count"].value_counts().sort_index())
print("\nPositive-label counts:")
print(eligible[labels].sum())
print("\nAll-zero rows:", int((eligible["label_count"] == 0).sum()))
PY
```

Review all-zero phishing-intention records before final evaluation.

---

## 27. Archive stale evaluation outputs

The earlier evaluator has no `--force` option and may reuse previous predictions. Archive old results before a fresh experiment:

```bash
mkdir -p outputs/archive_before_final_evaluation

find outputs \
  -type f \
  \( -name "predictions_*.jsonl" \
  -o -name "metrics_*.json" \
  -o -name "summary_*.csv" \
  -o -name "per_label_*.csv" \) \
  -exec mv {} \
  outputs/archive_before_final_evaluation/ \;
```

---

## 28. Test one positive ground-truth record

Create a temporary one-record manifest containing at least one positive label:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

labels = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

df = pd.read_csv("data/processed/manifest.csv").fillna("")

eligible = df[
    df["annotation_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("adjudicated")
    & df["phishing_status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("phishing")
].copy()

for label in labels:
    eligible[label] = pd.to_numeric(
        eligible[label],
        errors="coerce",
    ).fillna(0).astype(int)

eligible["label_count"] = eligible[labels].sum(axis=1)
positive = eligible[eligible["label_count"] > 0].head(1).copy()

if positive.empty:
    raise SystemExit("No positive adjudicated record exists.")

output = Path("data/processed/manifest_one_positive.csv")
positive.drop(columns=["label_count"]).to_csv(output, index=False)

print(positive[["sample_id", "image_path", *labels]].to_string(index=False))
print("Created:", output)
PY
```

Run:

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest_one_positive.csv \
  --mode gated \
  --limit 1
```

---

## 29. Run small evaluation checks

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  --limit 1
```

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  --limit 5
```

---

## 30. Run the full evaluations

### 30.1 Single-agent baseline

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode single
```

### 30.2 Always-on specialist ablation

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode always
```

### 30.3 Confidence-gated proposed pipeline

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated
```

If `run_ablation.py` matches the earlier evaluator’s argument interface, run:

```bash
python scripts/run_ablation.py \
  --manifest data/processed/manifest.csv
```

Check its interface first:

```bash
python scripts/run_ablation.py --help
```

---

## 31. Metrics

The evaluation reports:

- micro precision;
- micro recall;
- micro F1;
- exact subset accuracy;
- per-label precision;
- per-label recall;
- per-label F1;
- Accuracy by Complexity for one-, two- and three-intention records.

A `null` complexity score means the evaluated subset contained no record with that number of positive labels.

An all-zero metric output can occur when the selected record has no positive ground-truth labels or when predictions and ground truth do not overlap. Inspect the selected row before changing metric code.

---

## 32. Inspect result files

```bash
find outputs -type f | sort
```

Inspect the latest gated prediction where applicable:

```bash
tail -n 1 outputs/predictions_gated.jsonl
```

Display all metric files:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("outputs").rglob("metrics_*.json")):
    metrics = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + "=" * 72)
    print(path)
    print("=" * 72)
    print(json.dumps(metrics, indent=2))
PY
```

---

## 33. Freeze final experiment artefacts

Create an immutable snapshot:

```bash
mkdir -p experiments/final_annotation_v1
mkdir -p experiments/final_evaluation_v1
```

Copy annotation artefacts:

```bash
cp data/annotations/*.jsonl \
   experiments/final_annotation_v1/ \
   2>/dev/null || true

cp data/annotations/*.csv \
   experiments/final_annotation_v1/ \
   2>/dev/null || true

cp data/annotations/*.json \
   experiments/final_annotation_v1/ \
   2>/dev/null || true
```

Copy evaluation artefacts:

```bash
find outputs \
  -type f \
  -exec cp {} \
  experiments/final_evaluation_v1/ \;
```

Save the non-secret configuration:

```bash
grep -E \
  'OLLAMA_MODEL|CONFIDENCE_THRESHOLD|TOP_K|MAX_IMAGE_SIDE|OPENAI_ANNOTATION_MODEL|GEMINI_ANNOTATION_MODEL|GROQ_ADJUDICATION_MODEL' \
  .env \
  > experiments/final_evaluation_v1/config_summary.txt
```

Do not copy API keys.

---

## 34. Recommended end-to-end command sequence

```bash
# Enter and activate
cd "/Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM"
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q

# Generate manifest when needed
python scripts/prepare_data.py \
  --phish-iris data/raw/phish_iris \
  --putra data/raw/putra

# Independent annotation
python scripts/annotate_llm.py \
  --provider openai \
  --output-dir data/annotations

python scripts/annotate_llm.py \
  --provider gemini \
  --output-dir data/annotations

# Agreement
python scripts/calculate_annotation_agreement.py \
  --openai data/annotations/openai_annotations.jsonl \
  --gemini data/annotations/gemini_annotations.jsonl \
  --output-dir data/annotations \
  --create-image-links

# Groq tie-breaker
python scripts/groq_adjudicate.py \
  --sleep 20

# Create and complete human_review_queue.csv
# Then finalise
python scripts/finalise_annotations.py

# Apply ground truth
cp data/processed/manifest.csv \
   data/processed/manifest_before_adjudication.csv

python scripts/apply_adjudicated_labels.py \
  --manifest data/processed/manifest.csv \
  --adjudicated data/annotations/final_adjudicated.csv

# Validate one positive record
python scripts/evaluate.py \
  --manifest data/processed/manifest_one_positive.csv \
  --mode gated \
  --limit 1

# Full experiments
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

---

## 35. Troubleshooting

### `ModuleNotFoundError: phishintention`

```bash
python -m pip install -e .
```

Or temporarily:

```bash
PYTHONPATH="$PWD/src" python -m pytest -q
```

### Unsupported evaluator arguments

The earlier evaluator does not accept:

```text
--provider
--force
```

Use:

```bash
python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  --limit 1
```

### Project path changed

Regenerate `manifest.csv` because it may contain absolute paths. Back up annotations first.

### Gemini incomplete JSON

Rerun the same command. Successful sample IDs are skipped. Ensure retry logic and a sufficient maximum output-token setting are enabled.

### Groq rate limits

Use:

```bash
python scripts/groq_adjudicate.py \
  --sleep 20
```

The script reuses cached successful Groq responses.

### Groq malformed JSON

The hardened Groq adjudicator should recover a balanced JSON object where possible. Failures must be logged and marked for human review rather than terminating the whole run.

### All-zero evaluation metrics

Inspect the first selected ground-truth row. If all four labels are zero, Accuracy by Complexity will be `null` for one-, two- and three-label categories.

---

## 36. Research reporting requirements

Record the following in the final report:

- exact Qwen/Ollama model tag;
- quantisation where known;
- Ollama version;
- hardware and available memory;
- prompt versions;
- knowledge-base version;
- confidence threshold;
- top-k value;
- exact dataset subset;
- OpenAI annotation model;
- Gemini annotation model;
- Groq tie-breaker model;
- annotation agreement metrics;
- number of human-reviewed records;
- label distributions;
- single, always and gated evaluation results;
- limitations of machine-assisted ground truth;
- distinction between phishing detection and phishing-intention classification.

Recommended methodology wording:

> Two independent multimodal models generated initial screenshot annotations using the same label definitions and structured schema. OpenAI and Gemini did not receive each other’s responses or the evaluated Qwen prediction. Agreement was measured per label and over the complete label set. Groq was used only as a machine tie-breaker for disagreements. Every Groq-resolved or failed record was retained for human review. Only final adjudicated labels were applied to the manifest and used for the final Qwen evaluation.

---

## 37. Key limitations

- The project reproduces the paper’s architecture, not necessarily its exact runtime environment.
- TF-IDF retrieval is a project-specific implementation choice.
- The available datasets do not natively provide the four intention labels.
- Machine-assisted annotation is not equivalent to independent cybersecurity-expert annotation.
- Groq tie-breaker decisions must not be described as human adjudication.
- The intention pipeline should not be interpreted as a general phishing detector.
- Confidence values are model-generated scores and are not automatically calibrated probabilities.

---

## 38. Final checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Project installed with `pip install -e .`
- [ ] Ollama running
- [ ] Qwen model available
- [ ] Datasets extracted
- [ ] `manifest.csv` generated
- [ ] Image paths valid
- [ ] OpenAI annotations completed
- [ ] Gemini annotations completed
- [ ] Agreement metrics generated
- [ ] Groq tie-breaker completed
- [ ] Groq failures logged
- [ ] Human review queue completed
- [ ] `final_adjudicated.csv` generated
- [ ] Final labels applied to `manifest.csv`
- [ ] All-zero intent rows reviewed
- [ ] Stale evaluation outputs archived
- [ ] One-positive-record evaluation passed
- [ ] Single-agent evaluation completed
- [ ] Always-on evaluation completed
- [ ] Gated evaluation completed
- [ ] Final annotation artefacts frozen
- [ ] Final evaluation artefacts frozen
- [ ] Experimental limitations documented
