# PhishIntentionLLM

PhishIntentionLLM is a defensive screenshot-based multi-agent Retrieval-Augmented Generation system that predicts one or more intentions of a known phishing webpage:

- `credential_theft`
- `financial_fraud`
- `malware_distribution`
- `personal_information_harvesting`

The repository uses custom Python orchestration in `src/phishintention/pipeline.py`. It does not use LangChain or LangGraph.

## Responsible-use boundary

This project analyzes static screenshots for defensive research. It does not crawl live phishing websites, execute archived HTML, collect credentials, or generate phishing content. The four-intention pipeline receives screenshots already known to be phishing.

## Active dataset: Phish-Blitz screenshots

The active annotation dataset is a deduplicated, flattened collection of Phish-Blitz `online.png` screenshots. Every retained image is a known phishing screenshot.

Recommended repository layout:

```text
data/raw/phish_blitz/
├── images/
│   ├── phish_blitz_000001_....png
│   ├── phish_blitz_000002_....png
│   └── ...
└── source_manifest.csv          # Optional traceability from original paths
```

Generated files:

```text
data/processed/phish_blitz_manifest.csv
data/processed/phish_blitz_pending_manifest.csv
data/processed/manifest.csv
data/annotations/phish_blitz/
├── ollama_gemma_annotations.jsonl
├── ollama_gemma_failures.jsonl
├── ollama_minicpm_annotations.jsonl
├── ollama_minicpm_failures.jsonl
└── local/
    ├── agreement.csv
    ├── final_agreement.csv
    ├── human_review_queue.csv
    └── final_adjudicated.csv
```

Dataset rules:

- Use only captured screenshots. Do not access live phishing URLs.
- Keep one copy of every exact image after SHA-256 deduplication.
- Every active Phish-Blitz row has `source=phish_blitz` and `phishing_status=phishing`.
- Unannotated intention fields must remain empty. Do not initialize them to zero.
- `annotation_status=pending` means the image is ready for annotation.
- Stable sample IDs are derived from image-content hashes.
- Do not reuse annotations built from deleted duplicates or obsolete sample IDs.

## Architecture

```text
Screenshot
  -> Vision Analysis Agent
  -> TF-IDF retrieval from common threat KB
  -> Initial multi-label classifier
  -> Selected intention specialists
  -> Category-specific TF-IDF retrieval
  -> Validation and evidence synthesis
  -> Labels, confidence, evidence, and trace
```

Modes:

- `single`: single-agent baseline
- `always`: all four specialists
- `gated`: relevant specialists with confidence-aware validation

## Local annotation and evaluation design

```text
Gemma 3 12B + MiniCPM-V 8B
              |
       Agreement comparison
              |
       disputed labels only
              |
   Mistral Small 3.1 24B
              |
       Mandatory human review
              |
       final_adjudicated.csv
              |
        Apply to manifest.csv
              |
  Evaluate Qwen2.5-VL pipeline
```

Roles:

- `gemma3:12b`: primary local annotator A
- `minicpm-v:8b`: primary local annotator B
- `mistral-small3.1:24b`: provisional tie-breaker for disputed labels
- Human reviewer: final ground-truth authority
- `qwen2.5vl:72b`: evaluated model

Mistral must not overwrite labels on which the primary annotators agree. Every Mistral-resolved row remains marked `requires_human_review=True` until reviewed.

# First-time setup

## 1. Enter the project

```bash
cd "/Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM"
```

## 2. Create and activate the environment

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

## 3. Configure `.env`

```dotenv
OLLAMA_BASE_URL=http://localhost:11434

# Evaluated model
OLLAMA_MODEL=qwen2.5vl:72b

# Pipeline
CONFIDENCE_THRESHOLD=0.55
TOP_K=3
MAX_IMAGE_WIDTH=1280
MAX_IMAGE_HEIGHT=4096

# Primary annotators
OLLAMA_ANNOTATOR_1_NAME=gemma
OLLAMA_ANNOTATOR_1_MODEL=gemma3:12b
OLLAMA_ANNOTATOR_2_NAME=minicpm
OLLAMA_ANNOTATOR_2_MODEL=minicpm-v:8b

# Tie-breaker
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

## 4. Start Ollama and pull models

```bash
brew services start ollama
```

Alternatively:

```bash
ollama serve
```

Pull and verify models:

```bash
ollama pull gemma3:12b
ollama pull minicpm-v:8b
ollama pull mistral-small3.1:24b
ollama pull qwen2.5vl:72b
ollama list
curl -s http://localhost:11434/api/tags | python -m json.tool
```

Avoid keeping all large models loaded simultaneously.

# Dataset preparation

## 5. Remove exact duplicate image files

Dry run:

```bash
python scripts/remove_duplicate_phish_blitz_images.py \
  --images data/raw/phish_blitz/images \
  --report data/processed/phish_blitz_duplicate_images.csv
```

Review:

```bash
python - <<'PY'
import pandas as pd
p = "data/processed/phish_blitz_duplicate_images.csv"
df = pd.read_csv(p)
print("Duplicate groups:", df["duplicate_group"].nunique())
print("Files to delete:", (df["action"] == "delete").sum())
PY
```

Delete verified exact duplicates:

```bash
python scripts/remove_duplicate_phish_blitz_images.py \
  --images data/raw/phish_blitz/images \
  --report data/processed/phish_blitz_duplicate_images.csv \
  --delete
```

Verify:

```bash
python scripts/remove_duplicate_phish_blitz_images.py \
  --images data/raw/phish_blitz/images
```

Expected result: zero exact duplicate groups.

## 6. Regenerate the Phish-Blitz manifest

```bash
python scripts/build_phish_blitz_manifest.py \
  --project-root . \
  --images data/raw/phish_blitz/images \
  --output data/processed/phish_blitz_manifest.csv \
  --seed 42 \
  --train-ratio 0.70 \
  --validation-ratio 0.15
```

The remaining 15% is assigned to `test`.

Expected schema:

```text
sample_id,source,split,dataset_class,phishing_status,image_path,
relative_image_path,image_filename,brand,website_id,screenshot_variant,
annotation_status,annotator,notes,credential_theft,financial_fraud,
malware_distribution,personal_information_harvesting
```

The header must be exactly `brand`. Remove any accidental HTML markup around that column name.

## 7. Validate the generated manifest

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

p = "data/processed/phish_blitz_manifest.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)

missing = []
for value in df["image_path"]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        missing.append(str(path))

print("Rows:", len(df))
print("Unique sample IDs:", df["sample_id"].nunique())
print("Duplicate sample IDs:", df["sample_id"].duplicated().sum())
print("Duplicate image paths:", df["image_path"].duplicated().sum())
print("Missing images:", len(missing))
print("\nSplits:")
print(df["split"].value_counts(dropna=False))
print("\nStatuses:")
print(df["annotation_status"].value_counts(dropna=False))

assert len(df) == df["sample_id"].nunique()
assert not df["image_path"].duplicated().any()
assert not missing
print("\nVALIDATION PASSED")
PY
```

## 8. Activate the regenerated manifest

If Phish-Blitz is the only active dataset:

```bash
mkdir -p data/backups
cp data/processed/manifest.csv \
  "data/backups/manifest_before_phish_blitz_$(date +%Y%m%d_%H%M%S).csv" \
  2>/dev/null || true
cp data/processed/phish_blitz_manifest.csv data/processed/manifest.csv
```

Do not append an older Phish-Blitz manifest. It may reference deleted duplicate files or obsolete sample IDs.

## 9. Create the annotation-only manifest

The annotation script filters only `phishing_status == phishing`. A dedicated manifest prevents unrelated phishing rows from other sources from entering this run.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

source = Path("data/processed/manifest.csv")
target = Path("data/processed/phish_blitz_pending_manifest.csv")

df = pd.read_csv(source, dtype=str, keep_default_na=False)
subset = df[
    df["source"].str.strip().str.lower().eq("phish_blitz")
    & df["phishing_status"].str.strip().str.lower().eq("phishing")
    & df["annotation_status"].str.strip().str.lower().isin(["pending", "unlabelled"])
].copy()

missing = []
for index, row in subset.iterrows():
    path = Path(row["image_path"]).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        missing.append((row["sample_id"], str(path)))
    subset.at[index, "image_path"] = str(path)

if missing:
    pd.DataFrame(missing, columns=["sample_id", "image_path"]).to_csv(
        "data/processed/phish_blitz_missing_images.csv", index=False
    )
    raise RuntimeError("Missing images found")

if subset["sample_id"].duplicated().any():
    raise RuntimeError("Duplicate sample IDs found")

subset.to_csv(target, index=False)
print("Pending rows:", len(subset))
print("Output:", target)
PY
```

# Annotation workflow

The active annotation script supports:

```text
--provider-name
--model
--manifest
--output-dir
--limit
--force
```

Resume is automatic. Do not pass `--resume`. Do not pass `--output`, `--source`, or `--annotator`.

The commands below assume the script filename is `scripts/annotate_ollama.py`. If the repository file is named `scripts/annotate_local_ollama.py`, replace only the script filename and keep the arguments unchanged.

## 10. Create output directories

```bash
mkdir -p data/annotations/phish_blitz/smoke
mkdir -p data/annotations/phish_blitz/local
mkdir -p logs/phish_blitz
```

## 11. Gemma pilot

```bash
ollama stop minicpm-v:8b 2>/dev/null || true
ollama stop mistral-small3.1:24b 2>/dev/null || true
ollama stop qwen2.5vl:72b 2>/dev/null || true

python scripts/annotate_ollama.py \
  --provider-name gemma \
  --model gemma3:12b \
  --manifest data/processed/phish_blitz_pending_manifest.csv \
  --output-dir data/annotations/phish_blitz/smoke \
  --limit 10 \
  --force
```

Expected output:

```text
data/annotations/phish_blitz/smoke/ollama_gemma_annotations.jsonl
data/annotations/phish_blitz/smoke/ollama_gemma_failures.jsonl
```

## 12. MiniCPM pilot

```bash
ollama stop gemma3:12b 2>/dev/null || true

python scripts/annotate_ollama.py \
  --provider-name minicpm \
  --model minicpm-v:8b \
  --manifest data/processed/phish_blitz_pending_manifest.csv \
  --output-dir data/annotations/phish_blitz/smoke \
  --limit 10 \
  --force
```

## 13. Validate pilot output

```bash
python - <<'PY'
import json
from pathlib import Path

labels = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

for provider in ["gemma", "minicpm"]:
    path = Path(
        f"data/annotations/phish_blitz/smoke/"
        f"ollama_{provider}_annotations.jsonl"
    )
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"\n{provider}: {len(records)} records")
    for record in records[:3]:
        annotation = record.get("annotation", {})
        missing = [label for label in labels if label not in annotation]
        print(record.get("sample_id"), "missing labels:", missing)
        if missing:
            raise RuntimeError(f"{provider} output is missing labels")
print("\nPILOT VALIDATION PASSED")
PY
```

## 14. Run full Gemma annotation

```bash
ollama stop minicpm-v:8b 2>/dev/null || true
ollama stop mistral-small3.1:24b 2>/dev/null || true
ollama stop qwen2.5vl:72b 2>/dev/null || true

python scripts/annotate_ollama.py \
  --provider-name gemma \
  --model gemma3:12b \
  --manifest data/processed/phish_blitz_pending_manifest.csv \
  --output-dir data/annotations/phish_blitz \
  2>&1 | tee logs/phish_blitz/gemma.log
```

To resume after interruption, run the same command without `--force`:

```bash
python scripts/annotate_ollama.py \
  --provider-name gemma \
  --model gemma3:12b \
  --manifest data/processed/phish_blitz_pending_manifest.csv \
  --output-dir data/annotations/phish_blitz \
  2>&1 | tee -a logs/phish_blitz/gemma.log
```

## 15. Run full MiniCPM annotation

```bash
ollama stop gemma3:12b 2>/dev/null || true

python scripts/annotate_ollama.py \
  --provider-name minicpm \
  --model minicpm-v:8b \
  --manifest data/processed/phish_blitz_pending_manifest.csv \
  --output-dir data/annotations/phish_blitz \
  2>&1 | tee logs/phish_blitz/minicpm.log
```

Resume using the same command without `--force` and with `tee -a`.

## 16. Verify equal primary-annotator coverage

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

manifest = pd.read_csv(
    "data/processed/phish_blitz_pending_manifest.csv",
    dtype=str,
    keep_default_na=False,
)
expected = set(manifest["sample_id"])

def ids(path):
    path = Path(path)
    if not path.exists():
        return set()
    return {
        str(json.loads(line).get("sample_id", ""))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

gemma = ids("data/annotations/phish_blitz/ollama_gemma_annotations.jsonl")
minicpm = ids("data/annotations/phish_blitz/ollama_minicpm_annotations.jsonl")

print("Expected:", len(expected))
print("Gemma:", len(gemma))
print("MiniCPM:", len(minicpm))
print("Shared:", len(gemma & minicpm))
print("Missing Gemma:", len(expected - gemma))
print("Missing MiniCPM:", len(expected - minicpm))
print("Only Gemma:", len(gemma - minicpm))
print("Only MiniCPM:", len(minicpm - gemma))

if expected != gemma or expected != minicpm:
    raise RuntimeError("Primary annotator coverage is incomplete")
print("\nCOVERAGE VALIDATION PASSED")
PY
```

Rerunning an annotator without `--force` retries failures because only successful IDs are treated as completed.

## 17. Calculate provider-neutral agreement

```bash
python scripts/calculate_annotation_agreement.py \
  --annotation-a data/annotations/phish_blitz/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/phish_blitz/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --output-dir data/annotations/phish_blitz/local \
  --create-image-links
```

Expected output:

```text
data/annotations/phish_blitz/local/agreement.csv
```

## 18. Inspect agreement counts

```bash
python - <<'PY'
import pandas as pd
p = "data/annotations/phish_blitz/local/agreement.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
print("Rows:", len(df))
for column in [
    "agreement_status",
    "requires_adjudication",
    "requires_human_review",
    "image_quality_agreement",
]:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False))
PY
```

## 19. Run the Mistral tie-breaker pilot

```bash
ollama stop gemma3:12b 2>/dev/null || true
ollama stop minicpm-v:8b 2>/dev/null || true
ollama stop qwen2.5vl:72b 2>/dev/null || true

python scripts/adjudicate_local_ollama.py \
  --agreement data/annotations/phish_blitz/local/agreement.csv \
  --output data/annotations/phish_blitz/local/final_agreement.csv \
  --annotation-a data/annotations/phish_blitz/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/phish_blitz/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --model mistral-small3.1:24b \
  --limit 10 \
  --sleep 0
```

## 20. Run full Mistral adjudication

If the pilot wrote to `final_agreement.csv`, remove the pilot output only when the adjudication script does not resume safely. Otherwise, the full command may continue from its cache.

```bash
python scripts/adjudicate_local_ollama.py \
  --agreement data/annotations/phish_blitz/local/agreement.csv \
  --output data/annotations/phish_blitz/local/final_agreement.csv \
  --annotation-a data/annotations/phish_blitz/ollama_gemma_annotations.jsonl \
  --annotation-b data/annotations/phish_blitz/ollama_minicpm_annotations.jsonl \
  --provider-a gemma \
  --provider-b minicpm \
  --model mistral-small3.1:24b \
  --sleep 0 \
  2>&1 | tee logs/phish_blitz/mistral_adjudication.log
```

# Human review

## 21. Create the human-review queue

```bash
python - <<'PY'
import pandas as pd

source = "data/annotations/phish_blitz/local/final_agreement.csv"
target = "data/annotations/phish_blitz/local/human_review_queue.csv"

df = pd.read_csv(source, dtype=str, keep_default_na=False)
mask = (
    df["requires_human_review"]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes", "y"])
)
review = df[mask].copy()
review["human_review_completed"] = ""
review["human_review_notes"] = ""
review.to_csv(target, index=False)
print("Human-review rows:", len(review))
print("Output:", target)
PY
```

Open the queue:

```bash
open data/annotations/phish_blitz/local/human_review_queue.csv
```

Verify or correct all four `final_*` labels, then set:

```text
human_review_completed = 1
human_review_notes = concise screenshot-visible justification
```

## 22. Verify human-review completion

```bash
python - <<'PY'
import pandas as pd
p = "data/annotations/phish_blitz/local/human_review_queue.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
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
if (~completed).any():
    raise RuntimeError("Human review is incomplete")
PY
```

## 23. Finalize and apply labels

```bash
python scripts/finalise_annotations.py \
  --agreement data/annotations/phish_blitz/local/final_agreement.csv \
  --review data/annotations/phish_blitz/local/human_review_queue.csv \
  --output data/annotations/phish_blitz/local/final_adjudicated.csv
```

Back up and apply:

```bash
cp data/processed/manifest.csv \
  data/processed/manifest_before_local_adjudication.csv

python scripts/apply_adjudicated_labels.py \
  --manifest data/processed/manifest.csv \
  --adjudicated data/annotations/phish_blitz/local/final_adjudicated.csv
```

## 24. Validate applied ground truth

```bash
python - <<'PY'
import pandas as pd

p = "data/processed/manifest.csv"
df = pd.read_csv(p, dtype=str, keep_default_na=False)
active = df[df["source"].str.lower().eq("phish_blitz")].copy()
labels = [
    "credential_theft",
    "financial_fraud",
    "malware_distribution",
    "personal_information_harvesting",
]

print("Phish-Blitz rows:", len(active))
print("\nAnnotation status:")
print(active["annotation_status"].value_counts(dropna=False))

truthy = {"1", "true", "yes", "y"}
for label in labels:
    values = active[label].astype(str).str.strip().str.lower()
    print(
        f"{label}: positives={values.isin(truthy).sum()}, "
        f"empty={values.eq('').sum()}"
    )

if any(active[label].astype(str).str.strip().eq("").any() for label in labels):
    raise RuntimeError("Some final intention labels are empty")
PY
```

# Evaluation

The current evaluator supports `--manifest`, `--mode`, and `--limit`. Do not pass unsupported `--provider` or `--force` arguments.

## 25. Smoke-test evaluation

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

## 26. Run all evaluation modes

```bash
ollama stop gemma3:12b 2>/dev/null || true
ollama stop minicpm-v:8b 2>/dev/null || true
ollama stop mistral-small3.1:24b 2>/dev/null || true

python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode single \
  2>&1 | tee logs/phish_blitz/evaluation_single.log

python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode always \
  2>&1 | tee logs/phish_blitz/evaluation_always.log

python scripts/evaluate.py \
  --manifest data/processed/manifest.csv \
  --mode gated \
  2>&1 | tee logs/phish_blitz/evaluation_gated.log
```

The evaluator should use only known phishing rows with final adjudicated intention labels.

Expected metrics:

- Micro precision
- Micro recall
- Micro F1
- Per-intention precision, recall, F1, and support
- Exact subset accuracy
- Accuracy by complexity

# Troubleshooting

## Annotation script filename differs

If the file is named `scripts/annotate_local_ollama.py`, replace `scripts/annotate_ollama.py` in the commands. The supported arguments remain the same.

## Pilot produces no new rows

The smoke directory may already contain completed IDs. Either remove only the smoke JSONL files or rerun with `--force`.

## Full annotation resumes

Do not use `--force`. The script reads completed IDs from `ollama_<provider>_annotations.jsonl` and skips them automatically.

## Failure files keep growing

Failure logs are append-only. Rerunning retries unresolved sample IDs. Determine unresolved failures by subtracting successful IDs from failed IDs.

## All intention counts are zero

Stop before evaluation. Inspect the nested `annotation` object in both JSONL files and verify that all four decisions are present and parsed correctly. Empty fields must never be converted to zero before adjudication.

## Missing image paths

Regenerate `phish_blitz_manifest.csv` after moving or deleting images, then recreate `phish_blitz_pending_manifest.csv`.

## Ollama HTTP 500

```bash
tail -n 150 ~/.ollama/logs/server.log

grep -Ei "error|failed|panic|memory|alloc|runner|500|vision" \
  ~/.ollama/logs/server.log | tail -n 100
```

## Tall screenshots become unreadable

Preserve aspect ratio with separate maximum width and height limits. Recommended bounds are 1280 pixels wide and 4096 pixels high.

# Reproducibility checklist

- [ ] Python 3.11 and `.venv` configured
- [ ] Dependencies installed and tests pass
- [ ] `.env` configured
- [ ] Ollama and all four models available
- [ ] Flattened screenshots copied to `data/raw/phish_blitz/images`
- [ ] Exact duplicate images removed
- [ ] Fresh Phish-Blitz manifest generated
- [ ] Manifest validation passes
- [ ] Annotation-only manifest created
- [ ] Gemma and MiniCPM pilots succeed
- [ ] Equal primary-annotator coverage confirmed
- [ ] Provider-neutral agreement generated
- [ ] Mistral tie-breaker completed for disagreements
- [ ] Every tie-breaker row human-reviewed
- [ ] Final adjudicated labels applied to `manifest.csv`
- [ ] Single, always, and gated evaluations complete

## Recommended methodology wording

The project used a recent, deduplicated collection of captured Phish-Blitz webpage screenshots. Only known phishing screenshots entered the four-intention annotation workflow. Gemma 3 12B and MiniCPM-V 8B independently annotated each screenshot using the same label definitions and structured schema. Labels on which both primary annotators agreed were preserved. Mistral Small 3.1 24B produced provisional decisions only for disputed labels. Every Mistral-resolved row received human verification. Only final human-approved labels were applied to the manifest and used as ground truth for evaluating the Qwen2.5-VL multi-agent RAG architecture.
