# PhishIntentionLLM

PhishIntentionLLM is a defensive screenshot-based multi-agent Retrieval-Augmented Generation system that predicts one or more intentions of a known phishing webpage:

- `credential_theft`
- `financial_fraud`
- `malware_distribution`
- `personal_information_harvesting`

The repository uses custom Python orchestration in `src/phishintention/pipeline.py`. It does not use LangChain or LangGraph.

## Responsible-use boundary

This project analyses static screenshots for defensive research. It does not crawl live phishing websites, execute archived HTML, collect credentials, or generate phishing content. The four-intention pipeline should normally receive screenshots already known to be phishing. A normal login, payment, download, registration, or identity-verification screen is not sufficient by itself to prove malicious intent.

## Dataset update: the new dataset is the only active dataset

The project no longer uses the previous Phish-IRIS or Putra datasets. Remove those datasets from the active workspace after the migration backup is created.

The only active dataset is:

```text
Phishing dataset/
├── image/
│   ├── train/
│   │   ├── legitimate/
│   │   └── phishing/
│   ├── val/
│   │   ├── legitimate/
│   │   └── phishing/
│   └── test/
│       ├── legitimate/
│       └── phishing/
└── url/
```

Dataset rules:

- Preserve the provided `train`, `val`, and `test` splits.
- Each split contains `legitimate` and `phishing` classes.
- Only `phishing` screenshots enter the four-intention annotation workflow.
- `legitimate` screenshots remain in `manifest.csv` with `annotation_status=not_applicable` and all four intention labels set to `0`.
- The optional `url/` directory is preserved as metadata but is not consumed by the screenshot-only VLM pipeline.
- Stable sample IDs are derived from `phishing_dataset:<relative_image_path>`.
- Old manifests, annotations, agreement files, review queues, predictions, and metrics must not be reused because those artefacts reference the removed datasets and old sample IDs.

The active source name is:

```text
phishing_dataset
```

After generating the manifest, verify that no legacy source remains:

```bash
python - <<'PY2'
import pandas as pd

df = pd.read_csv("data/processed/manifest.csv").fillna("")
print(df["source"].value_counts(dropna=False))
unexpected = sorted(set(df["source"]) - {"phishing_dataset"})
if unexpected:
    raise SystemExit(f"Unexpected legacy sources: {unexpected}")
print("Manifest contains only the new dataset source.")
PY2
```

---

## Architecture

```text
Screenshot
  → Vision Analysis Agent
  → TF-IDF retrieval from common threat KB
  → Initial multi-label classifier
  → Selected intention specialists
  → Category-specific TF-IDF retrieval
  → Validation and evidence synthesis
  → Labels, confidence, evidence and trace
```

Modes:

- `single`: single-agent baseline
- `always`: all four specialists
- `gated`: relevant specialists with confidence-aware validation

## Local annotation and evaluation design

```text
Gemma 3 12B + MiniCPM-V 8B
              ↓
       Agreement comparison
              ↓ disputed labels only
   Mistral Small 3.1 24B
              ↓
       Mandatory human review
              ↓
       final_adjudicated.csv
              ↓
        Apply to manifest.csv
              ↓
  Evaluate Qwen2.5-VL pipeline
```

Roles:

- `gemma3:12b`: primary local annotator A
- `minicpm-v:8b`: primary local annotator B
- `mistral-small3.1:24b`: provisional tie-breaker for disputed labels
- human reviewer: final ground-truth authority
- `qwen2.5vl:72b`: evaluated model

Mistral must not overwrite labels on which the primary annotators agree. Every Mistral-resolved row remains `requires_human_review=True` until reviewed.

---

# First-time setup

## 1. Prerequisites

Install Python 3.11, Git, Ollama, and the new phishing screenshot dataset.

```bash
brew install ollama
```

## 2. Enter the project

```bash
cd "/Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM"
```

For a new clone:

```bash
git clone <repository-url>
cd PhishIntentionLLM
```

## 3. Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Validate:

```bash
which python
python -c "import sys; print(sys.executable)"
python -m compileall -q src scripts tests
python -m pytest -q
```

The fully local workflow needs packages including `pandas`, `numpy`, `Pillow`, `pydantic`, `python-dotenv`, `requests`, `scikit-learn`, `streamlit`, and `pytest`. OpenAI, Gemini, and Groq SDKs are not required by the active workflow.

## 4. Configure `.env`

```dotenv
OLLAMA_BASE_URL=http://localhost:11434

# Evaluated model
OLLAMA_MODEL=qwen2.5vl:72b

# Main pipeline
CONFIDENCE_THRESHOLD=0.55
TOP_K=3
MAX_IMAGE_WIDTH=1280
MAX_IMAGE_HEIGHT=4096

# Primary annotators
OLLAMA_ANNOTATOR_1_NAME=gemma
OLLAMA_ANNOTATOR_1_MODEL=gemma3:12b
OLLAMA_ANNOTATOR_2_NAME=minicpm
OLLAMA_ANNOTATOR_2_MODEL=minicpm-v:8b

# Provisional tie-breaker
OLLAMA_ADJUDICATOR_NAME=mistral
OLLAMA_ADJUDICATION_MODEL=mistral-small3.1:24b

# Request controls
OLLAMA_ANNOTATION_TIMEOUT=900
OLLAMA_ANNOTATION_MAX_RETRIES=2
OLLAMA_ADJUDICATION_TIMEOUT=900
OLLAMA_ADJUDICATION_MAX_RETRIES=2
OLLAMA_USE_JSON_SCHEMA=true
```

Do not commit `.env`.

## 5. Start Ollama and pull models

```bash
brew services start ollama
```

Or:

```bash
ollama serve
```

Pull models:

```bash
ollama pull gemma3:12b
ollama pull minicpm-v:8b
ollama pull mistral-small3.1:24b
ollama pull qwen2.5vl:72b
```

Verify:

```bash
ollama --version
ollama list
ollama ps
curl -s http://localhost:11434/api/tags | python -m json.tool
```

Avoid keeping all large models loaded simultaneously.

---

# New dataset

## 6. Required layout

The former Phish-IRIS and Putra datasets are removed from the active workflow. The project now uses:

```text
<dataset-root>/
├── image/
│   ├── train/
│   │   ├── legitimate/
│   │   └── phishing/
│   ├── val/
│   │   ├── legitimate/
│   │   └── phishing/
│   └── test/
│       ├── legitimate/
│       └── phishing/
└── url/                         # Optional metadata, not used for image inference
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, and `.gif`.

## 7. Migrate the repository

Copy and compile the migration script:

```bash
cp "/path/to/migrate_to_new_dataset.py" scripts/migrate_to_new_dataset.py
python -m py_compile scripts/migrate_to_new_dataset.py
```

Dry run:

```bash
python scripts/migrate_to_new_dataset.py \
  --project-root . \
  --dataset-root "/absolute/path/to/Phishing dataset" \
  --source-name phishing_dataset \
  --dry-run
```

Apply migration:

```bash
python scripts/migrate_to_new_dataset.py \
  --project-root . \
  --dataset-root "/absolute/path/to/Phishing dataset" \
  --source-name phishing_dataset \
  --create-dataset-symlink \
  --archive-outputs \
  --remove-old-raw
```

The migration:

- validates all six class folders;
- backs up the README, manifest, preparation script, annotations and optional outputs;
- replaces the old dataset-specific `prepare_data.py`;
- resets prior annotations because sample IDs no longer match;
- creates stable IDs from source name and relative image path;
- generates the new manifest;
- optionally removes the previous raw datasets;
- optionally links `data/raw/phishing_dataset` to the external dataset;
- writes `migration_report.json`.

Backups are stored under `migration_backups/`.

After migration, these legacy dataset folders must not remain active:

```text
data/raw/phish_iris
data/raw/putra
```

Verify removal:

```bash
for path in data/raw/phish_iris data/raw/putra; do
  if [ -e "$path" ]; then
    echo "Legacy dataset still exists: $path"
  fi
done
```

No output is expected.

## 8. Generate the manifest directly

```bash
python scripts/prepare_data.py \
  --dataset-root "/absolute/path/to/Phishing dataset" \
  --output data/processed/manifest.csv \
  --source-name phishing_dataset
```

Phishing rows receive:

```text
phishing_status = phishing
annotation_status = unlabelled
```

Legitimate rows receive:

```text
phishing_status = legitimate
annotation_status = not_applicable
credential_theft = 0
financial_fraud = 0
malware_distribution = 0
personal_information_harvesting = 0
```

Only known phishing screenshots enter the four-intention annotation workflow. Legitimate images are retained for reference or a separate phishing-detection experiment.

## 9. Validate the manifest

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/processed/manifest.csv").fillna("")
missing = df[~df["image_path"].map(lambda x: Path(str(x)).exists())]

print("Total records:", len(df))
print("Unique sample IDs:", df["sample_id"].nunique())
print("Unique image paths:", df["image_path"].nunique())
print("\nCounts by split and class:")
print(df.groupby(["split", "dataset_class"]).size().to_string())
print("\nMissing image paths:", len(missing))
print("Phishing annotation candidates:", int((df["annotation_status"] == "unlabelled").sum()))
print("Legitimate reference records:", int((df["annotation_status"] == "not_applicable").sum()))
PY
```

Required checks:

```text
Total records = Unique sample IDs
Total records = Unique image paths
Missing image paths = 0
```

---

# Git configuration

## 10. Update `.gitignore`

```bash
cp "/path/to/update_gitignore.py" scripts/update_gitignore.py
python scripts/update_gitignore.py --project-root . --dry-run
python scripts/update_gitignore.py --project-root .
```

The generated rules ignore raw or linked datasets, generated manifests, annotations, outputs, migration backups, experiment archives, secrets, virtual environments, and local model files. Source code, tests, README files, `knowledge/*.json`, `.env.example`, and `.gitkeep` placeholders remain trackable.

`.gitignore` does not untrack files that were committed earlier. Inspect and untrack generated artefacts:

```bash
git ls-files data/raw data/processed data/annotations outputs migration_backups experiments archive

git rm -r --cached \
  data/raw data/processed data/annotations outputs \
  migration_backups experiments archive \
  2>/dev/null || true

git add \
  data/raw/.gitkeep \
  data/processed/.gitkeep \
  data/annotations/.gitkeep \
  outputs/.gitkeep

git status --short
```

Validate representative rules:

```bash
git check-ignore -v \
  data/processed/manifest.csv \
  data/annotations/ollama_gemma_annotations.jsonl \
  data/annotations/local/final_adjudicated.csv \
  outputs/predictions_gated.jsonl \
  data/raw/phishing_dataset
```

---

# Knowledge base and UI

## 11. Validate the RAG KB

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("knowledge").glob("*.json")):
    entries = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.name}: {len(entries)} entries")
PY
```

The TF-IDF retrieval score ranks KB entries. It is not a classification probability.

## 12. Start Streamlit

Recommended:

```bash
python -m streamlit run app.py
```

Alternative:

```bash
streamlit run app.py
```

Because the annotation tab selects `annotation_status == "unlabelled"`, only phishing screenshots appear as pending. Legitimate screenshots are excluded because they are marked `not_applicable`.

---

# Local annotation implementation

## 13. Shared schemas

All providers must use the shared Pydantic models in:

```text
src/phishintention/annotators/schemas.py
```

Every annotation contains all four label decisions plus `image_quality`, `exclusion_reason`, and `annotation_notes`. Do not duplicate `LabelDecision` or `AnnotationOutput` classes in provider modules.

## 14. Generic Ollama annotator

`src/phishintention/annotators/ollama_annotator.py` must:

- use the shared schema;
- resize with separate maximum width and height;
- preserve aspect ratio for tall webpages;
- include `sample_id` in debug filenames;
- expose Ollama response bodies on HTTP failure;
- support JSON Schema mode and JSON fallback;
- validate every response with Pydantic.

Recommended bounds:

```text
maximum width = 1280
maximum height = 4096
```

## 15. Local annotation script

`scripts/annotate_ollama.py` supports:

```text
--provider-name
--model
--manifest
--output-dir
--limit
--force
```

Resume order:

```text
Load manifest
→ select phishing rows
→ load completed IDs
→ exclude completed rows
→ apply --limit
```

Expected files:

```text
data/annotations/ollama_gemma_annotations.jsonl
data/annotations/ollama_gemma_failures.jsonl
data/annotations/ollama_minicpm_annotations.jsonl
data/annotations/ollama_minicpm_failures.jsonl
```

## 16. Provider-neutral agreement

```bash
python scripts/calculate_annotation_agreement.py \
  --annotation-a data/annotations/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --output-dir data/annotations/local \
  --create-image-links
```

Expected output:

```text
data/annotations/local/agreement.csv
```

## 17. Dedicated local adjudication

Use `scripts/adjudicate_local_ollama.py`, not the old Groq retry script.

The script must:

1. load `agreement.csv`;
2. create missing final columns;
3. select label or image-quality disagreements;
4. preserve agreed labels;
5. ask Mistral only about disputed labels;
6. merge direct agreements and provisional decisions;
7. write `final_agreement.csv` without overwriting `agreement.csv`;
8. maintain a Mistral cache and deduplicated failure log;
9. set `requires_human_review=True` for Mistral-resolved rows.

Use provider-neutral provenance:

```text
local_primary_agreement
ollama_mistral_tiebreaker
ollama_mistral_failed
human_review
```

Use `adjudicator_model`, not `groq_model`.

---

# Run annotation

## 18. Gemma pilot

```bash
ollama stop minicpm-v:8b
ollama stop mistral-small3.1:24b
ollama stop qwen2.5vl:72b

python scripts/annotate_ollama.py \
  --provider-name gemma \
  --model gemma3:12b \
  --limit 10
```

## 19. MiniCPM pilot

```bash
ollama stop gemma3:12b

python scripts/annotate_ollama.py \
  --provider-name minicpm \
  --model minicpm-v:8b \
  --limit 10
```

## 20. Verify equal coverage

```bash
python - <<'PY'
import json
from pathlib import Path

def ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {json.loads(line)["sample_id"] for line in path.read_text().splitlines() if line.strip()}

gemma = ids("data/annotations/ollama_gemma_annotations.jsonl")
minicpm = ids("data/annotations/ollama_minicpm_annotations.jsonl")
print("Gemma:", len(gemma))
print("MiniCPM:", len(minicpm))
print("Shared:", len(gemma & minicpm))
print("Only Gemma:", len(gemma - minicpm))
print("Only MiniCPM:", len(minicpm - gemma))
PY
```

## 21. Full primary annotation

```bash
ollama stop minicpm-v:8b
ollama stop mistral-small3.1:24b
ollama stop qwen2.5vl:72b
python scripts/annotate_ollama.py --provider-name gemma --model gemma3:12b

ollama stop gemma3:12b
python scripts/annotate_ollama.py --provider-name minicpm --model minicpm-v:8b
```

## 22. Calculate agreement

```bash
python scripts/calculate_annotation_agreement.py \
  --annotation-a data/annotations/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --output-dir data/annotations/local \
  --create-image-links
```

## 23. Mistral adjudication

```bash
ollama stop gemma3:12b
ollama stop minicpm-v:8b
ollama stop qwen2.5vl:72b
```

Pilot:

```bash
python scripts/adjudicate_local_ollama.py \
  --agreement data/annotations/local/agreement.csv \
  --output data/annotations/local/final_agreement.csv \
  --annotation-a data/annotations/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --model mistral-small3.1:24b \
  --limit 10 \
  --sleep 0
```

Full run:

```bash
python scripts/adjudicate_local_ollama.py \
  --agreement data/annotations/local/agreement.csv \
  --output data/annotations/local/final_agreement.csv \
  --annotation-a data/annotations/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --model mistral-small3.1:24b \
  --sleep 0
```

---

# Human approval

## 24. Generate the review queue

```bash
python - <<'PY'
import pandas as pd

source = "data/annotations/local/final_agreement.csv"
target = "data/annotations/local/human_review_queue.csv"
df = pd.read_csv(source).fillna("")
mask = df["requires_human_review"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
review = df[mask].copy()
review["human_review_completed"] = ""
review["human_review_notes"] = ""
review.to_csv(target, index=False)
print("Human-review rows:", len(review))
PY
```

Open:

```bash
open data/annotations/local/human_review_queue.csv
```

Verify or correct all four `final_*` labels, then set:

```text
human_review_completed = 1
human_review_notes = concise screenshot-visible justification
```

## 25. Verify completion

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/annotations/local/human_review_queue.csv").fillna("")
completed = df["human_review_completed"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
print("Total:", len(df))
print("Completed:", int(completed.sum()))
print("Incomplete:", int((~completed).sum()))
PY
```

## 26. Finalise and apply labels

```bash
python scripts/finalise_annotations.py \
  --agreement data/annotations/local/final_agreement.csv \
  --review data/annotations/local/human_review_queue.csv \
  --output data/annotations/local/final_adjudicated.csv

cp data/processed/manifest.csv \
  data/processed/manifest_before_local_adjudication.csv

python scripts/apply_adjudicated_labels.py \
  --manifest data/processed/manifest.csv \
  --adjudicated data/annotations/local/final_adjudicated.csv
```

---

# Evaluation

## 27. Evaluator interface

```bash
python scripts/evaluate.py --help
```

The current evaluator supports `--manifest`, `--mode`, and `--limit`. Do not pass unsupported `--provider` or `--force` arguments.

## 28. Run evaluation

```bash
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode gated --limit 1
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode gated --limit 5
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode single
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode always
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode gated
```

The evaluator should use only phishing rows with final adjudicated intention labels. Legitimate rows are outside the four-intention task unless a separate phishing-detection experiment is implemented.

Report micro precision, micro recall, micro F1, per-intention precision/recall/F1/support, exact subset accuracy, and Accuracy by Complexity.

---

# Troubleshooting

## Missing dataset folders

The dataset must contain all six folders under `image/{train,val,test}/{legitimate,phishing}`.

## Missing image paths

Regenerate `manifest.csv` after moving the external dataset.

## Pilot produces no new rows

Ensure completed IDs are excluded before applying `--limit`.

## Tall screenshots become unreadable

Preserve aspect ratio with separate width and height limits. Do not force a square thumbnail.

## Mistral changes agreed labels

Preserve primary agreement and use Mistral only for disputed labels.

## Ollama HTTP 500

```bash
tail -n 150 ~/.ollama/logs/server.log

grep -Ei "error|failed|panic|memory|alloc|runner|500|vision" \
  ~/.ollama/logs/server.log | tail -n 100
```

## `.gitignore` appears ineffective

The file may already be tracked. Remove generated files from the Git index with `git rm --cached`, then re-add `.gitkeep` placeholders.

---

# Reproducibility checklist

- [ ] Python 3.11 and `.venv` configured
- [ ] dependencies installed and tests pass
- [ ] `.env` configured
- [ ] Ollama and all four models available
- [ ] new dataset has all six class folders
- [ ] migration dry run succeeds
- [ ] old dataset artefacts archived
- [ ] new manifest generated and validated
- [ ] `.gitignore` updated and generated files untracked
- [ ] KB validated
- [ ] Streamlit starts
- [ ] Gemma and MiniCPM pilots succeed
- [ ] equal coverage confirmed
- [ ] provider-neutral agreement generated
- [ ] Mistral pilot and full run complete
- [ ] every tie-breaker row human-reviewed
- [ ] final adjudicated labels applied to manifest
- [ ] single, always, and gated evaluations complete

## Recommended methodology wording

> The project used one screenshot dataset organised into train, validation and test splits with legitimate and phishing class folders. Only known phishing screenshots entered the four-intention annotation workflow. Gemma 3 12B and MiniCPM-V 8B independently annotated each phishing screenshot using the same label definitions and structured schema. Labels on which both primary annotators agreed were preserved. Mistral Small 3.1 24B produced provisional decisions only for disputed labels. Every Mistral-resolved row received human verification. Only final human-approved labels were applied to the manifest and used as ground truth for evaluating the Qwen2.5-VL multi-agent RAG architecture.
