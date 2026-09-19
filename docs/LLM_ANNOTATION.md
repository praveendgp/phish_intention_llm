# Dual-LLM annotation

## Models

Pilot configuration:

- OpenAI: `gpt-4.1-mini-2025-04-14`
- Gemini: `gemini-3.8-flash`

Quality escalation:

- OpenAI: `gpt-4.1-2025-04-14`

The evaluated project model remains Qwen2.5-VL. Do not expose Qwen predictions to either annotator.

## Install

```bash
python -m pip install "openai>=1.0" "google-genai>=2,<3"
```

Add API credentials and model settings to `.env` or the shell environment.

## Pilot

```bash
python scripts/annotate_llm.py --provider openai --limit 10
python scripts/annotate_llm.py --provider gemini --limit 10
python scripts/calculate_annotation_agreement.py
```

## Full run

```bash
python scripts/annotate_llm.py --provider openai
python scripts/annotate_llm.py --provider gemini
python scripts/calculate_annotation_agreement.py
```

The scripts append JSONL and skip completed sample IDs. Use `--force` only when deliberately replacing a run after changing a model or prompt. Preserve old runs under versioned experiment folders.

## Human adjudication

Review `data/annotations/adjudication_queue.csv`. Create `data/annotations/final_adjudicated.csv` with:

```text
sample_id,credential_theft,financial_fraud,malware_distribution,personal_information_harvesting,adjudication_notes
```

Then apply final labels:

```bash
python scripts/apply_adjudicated_labels.py
```

Only `adjudicated` rows should be used for final evaluation.
