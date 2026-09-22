# 🎣 PhishIntentionLLM

**Uncovering Phishing Website Intentions through Multi-Agent Retrieval-Augmented Generation**

A complete implementation of *PhishIntentionLLM* (Li, Manickam, Chong & Karuppayah — ICDF2C 2025), adapted to run entirely on **local open-source vision-language models served by Ollama**.

Ordinary phishing detectors answer *“is this page phishing?”*. This project answers ***“what does the attacker actually want?”*** across four intentions:

| | Intention | What the attacker is after |
|---|---|---|
| 🔑 | **Credential Theft** | usernames, passwords, OTPs |
| 💳 | **Financial Fraud** | card data, transfers, payments |
| 🦠 | **Malware Distribution** | getting a payload executed |
| 🪪 | **Personal Information Harvesting** | identity data, KYC documents |

---

## 📖 Read this first: how the project works

There are **two independent stages**, and the order matters. This is the single most important thing to understand before running anything.

```
                    ┌──────────────── STAGE A · MANIFEST ────────────────┐
  screenshot ──────►│  Annotator A (VLM) ─┐                              │
                    │                      ├─► tie-break ─► finaliser    │──┐
                    │  Annotator B (VLM) ─┘   (only on dispute)          │  │
                    └───────────────────────────────────────────────────┘  │
                              writes  data/outputs/manifest.jsonl           │
                                                                             │ reference
                    ┌──────────── STAGE B · FRAMEWORK (5 layers) ───────┐   │ labels
  screenshot ──────►│ 👁️ Vision → 🧠 Context → 🏷️ Classify → 🔬 Experts → ✅ │──┤
                    └───────────────────────────────────────────────────┘   │
                            writes  data/outputs/predictions.jsonl           │
                                                                             ▼
                                                                   📊 EVALUATION
```

**Stage A — the annotators build the answer key.** Two different vision models label each screenshot independently. When they disagree, a third model breaks the tie. A fourth signs off the final label. The result, `manifest.jsonl`, is the **reference** you will measure against.

**Stage B — the framework takes the exam.** The five layers from the paper analyse the same screenshots and write `predictions.jsonl`. The framework **never reads the manifest** — that separation is what makes the evaluation meaningful.

**Evaluation scores B against A.**

> **Every agent in both stages is a vision-language model that receives the screenshot itself** — never a text description of it. This is enforced in code: an agent whose model isn't multimodal refuses to start.

---

## 1 · Install

```bash
unzip PhishIntentionLLM.zip -d PhishIntentionLLM
cd PhishIntentionLLM

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.10 or newer.

## 2 · Start Ollama and pull the models

Leave this running in its own terminal:

```bash
ollama serve
```

In a second terminal:

```bash
ollama pull minicpm-v:8b
ollama pull granite3.2-vision:2b
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
```

Roughly 25 GB total. All four are multimodal, none are from the paper, none are LLaMA-family.

**Under 12 GB VRAM?** Use the light profile in [§9](#9--low-vram-profile) instead — pull `moondream` and `gemma3:4b` and skip the two large models.

## 3 · Add the datasets

Both sources keep their **original folder structure** — nothing needs renaming:

```
data/raw/putra/
  phishing/<record-id>/screenshots/*.png
                      /assets/            (optional: url.txt, meta.json)
  not-phishing/<record-id>/screenshots/*.png

data/raw/phishIris/
  train/<brand>/*.png       # apple, amazon, chase, dhl, paypal, … other
  val/<brand>/*.png
```

Full details in `data/raw/README.md`. No datasets yet? You can still try a single screenshot via the Analyse page.

## 4 · Verify everything

```bash
python scripts/check_setup.py
```

This checks the knowledge base, both dataset paths, Ollama connectivity, that all nine model roles are pulled, **and that each one is genuinely multimodal**. Fix anything it flags before continuing — a green run here saves a lot of confusion later.

---

## 5 · Your first run (start small)

Do this before committing to a large batch. It takes about 10 minutes and proves the whole loop works end to end.

```bash
# Stage A · annotators label 5 screenshots
python scripts/build_manifest.py --n 5

# Stage B · framework predicts on those same 5
python scripts/run_predictions.py --n 5 --scope manifest

# Score the framework against the manifest
python scripts/run_evaluation.py
```

`--scope manifest` restricts the run to samples that already have reference labels, so results are immediately scoreable.

### Reading the manifest output

```
[  1/5] TB  putra::phishing::640b9757…  -> Credential Theft            (partial agreement, 66.7s)
[  2/5] OK  phishiris::train::paypal…   -> Credential Theft, Financial Fraud (full agreement, 70.5s)
```

| Flag | Meaning |
|---|---|
| `OK ` | The two annotators settled it themselves |
| `TB ` | They disputed a category — the **tie-breaker** was invoked to rule on it |
| `ERR` | Annotation failed for that sample |

Seeing plenty of `TB` early on is healthy: it means your two annotators are genuinely independent. If *everything* is `TB`, they're diverging too much to produce a trustworthy reference.

### Reading the evaluation output

```
=== Evaluation vs the manifest (5 samples) ===
  Precision (micro) : 0.8571
  Recall    (micro) : 0.8571
  F1        (micro) : 0.8571
  Overall accuracy  : 0.8000
```

**Precision** — of the intentions the framework claimed, how many the annotators agreed with.
**Recall** — of the intentions the annotators found, how many the framework caught.
**Overall accuracy** — how often the framework's label *set* matched exactly (strict: partial credit gets none).

---

## 6 · Scaling up

Once the small run looks right:

```bash
# Stage A · build a proper reference set (plan ~60-90s per sample)
python scripts/build_manifest.py --n 100 --csv data/outputs/manifest.csv

# Stage B · framework
python scripts/run_predictions.py --n 100 --scope manifest

# Stage B · single-agent baseline, for the comparison in your report
python scripts/run_predictions.py --n 100 --scope manifest --mode single

# Evaluate, with a per-sample error breakdown
python scripts/run_evaluation.py --errors
```

Both scripts skip work already done, so you can stop and resume freely:

```bash
python scripts/build_manifest.py  --n 200                 # only annotates new samples
python scripts/run_predictions.py --n 200 --scope manifest --skip-done
```

### Check your manifest is sound

If an annotator ever crashes, that row falls back to a single opinion with no cross-check — and it still counts as reference truth. Audit for this:

```bash
python -c "
import json
rows=[json.loads(l) for l in open('data/outputs/manifest.jsonl')]
solo=[r for r in rows if r['agreement']=='single annotator']
print(f'{len(solo)}/{len(rows)} rows have only one annotator')
"
```

Re-annotate any you find:

```bash
python scripts/build_manifest.py --n 100 --overwrite
```

---

## 7 · The UI

```bash
./run_ui.sh          # Windows: run_ui.bat
```

Open **http://localhost:8501**.

| # | Page | What you do there |
|---|---|---|
| 1 | 📚 **Dataset Explorer** | Confirm both sources load; see which samples have manifest labels |
| 2 | 📋 **Manifest Builder** | Run the annotators; inspect every row — votes, agreement, tie-break reasoning; export CSV |
| 3 | ✍️ **Manual Annotation** | Two engineers + reviewer protocol; sign off ⚠️ flagged manifest rows |
| 4 | 🤖 **Framework Predictions** | Batch-run the framework or baseline over manifest-covered samples |
| 5 | 📊 **Evaluation** | Micro metrics, Acc_comp, per-class table, error analysis, run comparison |
| — | 🎣 **Analyse** | Upload a single screenshot and watch the agents work — best for demos |
| — | 🧠 **Knowledge Base** | Inspect K_B and K_c; test retrieval queries |
| — | ⚙️ **Settings** | Ollama health, both model groups, multimodal capability audit |

The agent flow animates live, grouped by stage, with each node showing its model and conclusion:

```
MANIFEST STAGE
🗳️  Annotator A            [VLM · sees image]   Labelled: CT 0.92            ● Done
🗳️  Annotator B            [VLM · sees image]   Labelled: CT 0.92, PIH 0.55  ● Done
⚖️  Tie-Breaker            [VLM · sees image]   Ruled on 1; kept: none       ● Done
📝  Manifest Finaliser     [VLM · sees image]   Labels: CT (quality: high)   ● Done

FRAMEWORK STAGE
👁️  Vision Analysis Agent  [VLM · sees image]   Saw 2 form fields, password  ● Done
🧠  Context Enrichment     [VLM · sees image]   Tagged 1, +1 missed element  ● Done
🏷️  Classification Agent   [VLM · sees image]   Nominated CT 0.93            ● Done
🔬  Credential Theft Expert[VLM · sees image]   CONFIRMED CT @ 0.93          ● Done
🔬  Financial Fraud Expert                      Not nominated                ◌ Skipped
✅  Validation & Synthesis [VLM · sees image]   Final: CT 0.93               ● Done
```

---

## 8 · Optional: human ground truth

The manifest is machine-generated. For a stronger claim in your report, add human labels using the paper's protocol — two engineers label independently, a reviewer signs off:

1. **✍️ Manual Annotation** → select your identity → label samples → repeat as the second engineer → then as `reviewer`.
2. Export and evaluate against it:

```bash
python scripts/build_ground_truth.py
python scripts/run_evaluation.py --reference ground_truth
```

You can also have the reviewer verify machine-generated manifest rows in place (queue: *⚠️ Manifest flagged for review*), then score against only those:

```bash
python scripts/run_evaluation.py --verified-only
```

---

## 9 · Low-VRAM profile

Edit `config.yaml` — keep every model multimodal:

```yaml
models:
  annotator_a:         {model: "moondream",            vision: true, temperature: 0.1}
  annotator_b:         {model: "granite3.2-vision:2b", vision: true, temperature: 0.1}
  manifest_tiebreaker: {model: "gemma3:4b",            vision: true, temperature: 0.0}
  manifest_finalizer:  {model: "gemma3:4b",            vision: true, temperature: 0.0}
  vision:              {model: "moondream",            vision: true, temperature: 0.1}
  context:             {model: "gemma3:4b",            vision: true, temperature: 0.0}
  classifier:          {model: "gemma3:4b",            vision: true, temperature: 0.0}
  specialist:          {model: "gemma3:4b",            vision: true, temperature: 0.0}
  validator:           {model: "gemma3:4b",            vision: true, temperature: 0.0}

ollama:
  num_ctx: 4096
framework:
  max_image_edge: 896
```

Keep the two annotators on **different model families** — that diversity is what makes the agreement signal meaningful.

---

## 10 · Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `token repeat limit reached` | Small VLM loops under Ollama's JSON grammar. Add the model to `ollama.disable_json_format_for`, or set `models.<role>.json_format: false` |
| `exceeds the available context size` | Prompt + image too large. Lower `framework.max_image_edge` to `896` — image tokens dominate. Or raise `models.<role>.num_ctx` |
| `(single annotator)` in the log | An annotator crashed; that row has no cross-check. Re-run with `--overwrite` |
| `VisionCapabilityError` at startup | A role points at a text-only model — swap it in `config.yaml` |
| Everything labelled Credential Theft | Raise `framework.confidence_threshold`; strengthen the `I_c` indicators for the other categories in `specialist_kb.json` |
| Annotators always agree (IAA = 100%) | Your two annotator models are too similar — pick more distinct architectures |
| Tie-breaker never fires | Lower `manifest.agreement_margin` (a solo claim escalates when its confidence < 1 − margin) |
| Too slow | Smaller models, lower `max_image_edge`, or set `framework.enable_feedback_loop: false` |
| Out of VRAM | Shrink `ollama.num_ctx`, use 2–4B VLMs, reduce `ollama.keep_alive` |

### Test without Ollama

```bash
python tests/test_pipeline.py          # metrics, consensus, KB, manifest store
python tests/test_end_to_end_mock.py   # both stages + a VLM-enforcement audit
```

The second prints an audit proving every agent call carried the screenshot.

---

## 11 · Architecture reference

**Framework (Fig. 3 of the paper)**

```
Screenshot (encoded once, shared by every agent)
    ├─ Layer 1  Vision Analysis ....... OCR, form fields, buttons, branding, layout
    ├─ Layer 2  Context Enrichment .... tags elements with K_B patterns (RAG)
    ├─ Layer 3  Classification ........ scores all 4 categories, Top-k(S, k≤3)
    ├─ Layer 4  Specialist Analysis ... 4 expert agents grounded in K_c
    └─ Layer 5  Validation & Synthesis  evidence chains, conflicts, τ filter
                  └── feedback loop if confidence < τ → widen to remaining categories
```

**Knowledge architecture (Definition 1)** — `K_B = {P_c, P_v, P_t, P_u}` and `K_c = {F_c, D_c}` with `D_c = {T_c, M_c, I_c}`, both in `data/knowledge_base/*.json`. Retrieval uses a dependency-free TF-IDF index by default; set `framework.embedding_model` for sentence-transformers.

**Model roles**

| Stage | Role | Default model |
|---|---|---|
| A | `annotator_a` | `minicpm-v:8b` |
| A | `annotator_b` | `granite3.2-vision:2b` |
| A | `manifest_tiebreaker` | `mistral-small3.2:24b` |
| A | `manifest_finalizer` | `gemma3:12b` |
| B | `vision` | `minicpm-v:8b` |
| B | `context` / `classifier` / `specialist` | `gemma3:12b` |
| B | `validator` | `mistral-small3.2:24b` |

**Metrics implemented**

| Metric | Equation in paper |
|---|---|
| Overall accuracy (exact set match) | Eq. 3 |
| Precision / Recall / F1 (micro) | Eq. 4–6 |
| Accuracy by complexity `Acc_comp(k)`, `t_k = 1,1,2` | Eq. 7–8 |
| Per-class P / R / F1 / Accuracy | Eq. 9–10 |

Plus set-level agreement, the credential-theft benchmark, co-occurrence and sector × intention analyses, and a reference-quality panel (inter-annotator agreement, tie-break rate, review backlog).

**Project layout**

```
PhishIntentionLLM/
├── config.yaml                  # models (2 groups), τ, datasets, storage
├── run_ui.sh / run_ui.bat
├── app/                         # Streamlit console (8 pages)
├── src/phishintentionllm/
│   ├── agents/                  # base (VLM enforcement) + the 5 framework layers
│   ├── annotation/              # annotators, manifest_builder, manual, store
│   ├── datasets/                # putra, phishiris, registry
│   ├── rag/                     # knowledge_base, retriever
│   ├── pipeline/                # orchestrator, single_agent, batch
│   ├── evaluation/              # metrics (Eq. 3-10), evaluator
│   ├── llm/                     # vision-enforcing Ollama client
│   └── utils/                   # image encoding, tolerant JSON parsing, logging
├── data/
│   ├── knowledge_base/          # K_B and K_c
│   └── outputs/                 # manifest.jsonl · predictions.jsonl · runs/
├── scripts/                     # check_setup, build_manifest, run_predictions, …
└── tests/                       # offline unit tests + mocked end-to-end test
```

See `docs/ARCHITECTURE.md` for the full paper-to-code mapping.

---

## 12 · Command reference

```bash
# setup
python scripts/check_setup.py

# Stage A · manifest
python scripts/build_manifest.py --n 100
python scripts/build_manifest.py --n 100 --overwrite          # re-annotate
python scripts/build_manifest.py --n 50 --sources putra       # one source only
python scripts/build_manifest.py --n 50 --csv out.csv         # + CSV export

# Stage B · predictions
python scripts/run_predictions.py --n 100 --scope manifest
python scripts/run_predictions.py --n 100 --scope manifest --mode single
python scripts/run_predictions.py --n 100 --scope random --skip-done

# evaluation
python scripts/run_evaluation.py
python scripts/run_evaluation.py --errors
python scripts/run_evaluation.py --verified-only
python scripts/run_evaluation.py --reference ground_truth
python scripts/run_evaluation.py --compare \
    data/outputs/runs/run_<framework_id>.json \
    data/outputs/runs/run_<baseline_id>.json

# human ground truth
python scripts/build_ground_truth.py

# UI
./run_ui.sh
```

---

## 📚 Reference

Li, W., Manickam, S., Chong, Y.-W., & Karuppayah, S. (2025). *PhishIntentionLLM: Uncovering Phishing Website Intentions through Multi-Agent Retrieval-Augmented Generation.* ICDF2C 2025. arXiv:2507.15419

Datasets: Putra phishing website dataset (Zenodo 8041387) · Phish-IRIS (Dalgic, Bozkir & Aydos, ISMSIT).

> Research and educational use only. Handle phishing screenshots in an isolated environment.
