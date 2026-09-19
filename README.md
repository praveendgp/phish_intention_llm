# PhishIntentionLLM Two-Dataset Reproduction

Implementation of the five-layer workflow in the base paper:

1. Phish-IRIS: https://data.mendeley.com/datasets/tk4fkswtj5/1
2. Putra Phishing Website Dataset: https://zenodo.org/records/8041387

## Important research limitation
Neither source provides the paper's four intention ground-truth labels. Phish-IRIS provides brand/other folders; Putra provides phishing/legitimate status, brand and metadata. Therefore, this project does **not** pretend those labels exist. It provides:

- adapters that discover screenshots from both datasets;
- a reproducible annotation manifest and Streamlit labelling screen;
- single-agent and five-layer multi-agent inference;
- dual knowledge-base retrieval;
- confidence-gated specialists and feedback loop;
- evaluation only on human-labelled records;
- paper metrics: micro precision/recall/F1, subset accuracy and Accuracy by Complexity;
- an ablation runner for single-agent, always-on multi-agent and gated multi-agent.

## Architecture
`screenshot -> vision/OCR -> basic KB retrieval -> initial classifier -> 4 specialists -> validator -> multi-label result`

The four labels are `credential_theft`, `financial_fraud`, `malware_distribution`, and `personal_information_harvesting`.

## Quick start, macOS Apple Silicon

```bash
brew install python@3.11 tesseract ollama
ollama serve
ollama pull qwen2.5vl:3b
cd PhishIntentionLLM_two_datasets
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Download and extract the datasets yourself, preserving their licence/terms:

```text
data/raw/phish_iris/   # extracted Mendeley download
data/raw/putra/        # extracted Zenodo phishing zip(s); metadata CSV optional
```

Do not download all 38.5 GB from Zenodo initially. Start with `phishing.csv` plus one phishing archive such as `phishing_0001-0500.zip`, extract it under `data/raw/putra/`, and scale later.

## Run the pipeline

```bash
python scripts/prepare_data.py --phish-iris data/raw/phish_iris --putra data/raw/putra
streamlit run app.py
```

In the app, use **Annotate dataset** to assign one or more intentions. For defensible evaluation, use two independent annotators and an adjudicated final manifest. See `docs/ANNOTATION_GUIDE.md`.

Analyse one screenshot:

```bash
python scripts/predict.py --image path/to/screenshot.png --mode gated
```

Smoke test without Ollama:

```bash
python scripts/predict.py --image tests/fixtures/sample_login.png --provider mock
pytest -q
```

Evaluate labelled data:

```bash
python scripts/evaluate.py --manifest data/processed/manifest.csv --mode gated
python scripts/run_ablation.py --manifest data/processed/manifest.csv --limit 100
```

Outputs are written to `outputs/` as JSONL/CSV/JSON. Resume is supported by skipping existing sample IDs.

## Dataset manifest schema

`sample_id, source, image_path, brand, phishing_status, split, credential_theft, financial_fraud, malware_distribution, personal_information_harvesting, annotation_status, annotator, notes`

Only rows with `annotation_status=adjudicated` or `labelled` enter metric computation.

## Reproduction statement
This is an architectural reproduction, not an exact numerical reproduction. The paper's released four-intention labels are unavailable in the two selected source datasets, and the paper used large commercial VLMs. Report your newly annotated sample size, agreement, model, prompts, threshold, and dataset subset alongside results.

## Safety
Use released static screenshots only. The repository contains no crawler, credential collector, browser automation against live sites, exploit code or phishing-page generator.
