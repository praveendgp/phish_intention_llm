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

> ⚠️ **The manifest is machine-generated.** Your metrics measure agreement with your annotator ensemble, not with human judgement. Always state this when reporting results, and validate the manifest before tuning anything — see [§7](#7--validate-the-manifest-before-you-tune-anything).

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
ollama pull gemma3:4b
ollama pull mistral-small3.2:24b
```

All multimodal, none from the paper, none LLaMA-family.

> **Why not `granite3.2-vision:2b`?** It was the original Annotator B but proved unstable in practice — context overflows, empty responses and token-repeat loops. Each failure silently degrades a manifest row to `(single annotator)` with no cross-check. `gemma3:4b` is the recommended replacement: still multimodal, still a different family from MiniCPM-V, far steadier.

**Under 12 GB VRAM?** See the light profile in [§10](#10--low-vram-profile).

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

Full details in `data/raw/README.md`.

## 4 · Recommended configuration

Apply these before your first real run. Each one fixes a problem encountered in practice.

```yaml
models:
  annotator_a:
    model: "minicpm-v:8b"          # OpenBMB
    vision: true
    temperature: 0.1

  annotator_b:
    model: "gemma3:4b"             # Google - different family from A
    vision: true
    temperature: 0.2

  manifest_tiebreaker:
    model: "mistral-small3.2:24b"  # Mistral - independent of both annotators
    vision: true
    temperature: 0.0

  manifest_finalizer:
    model: "mistral-small3.2:24b"  # same model is fine: arbitration, not voting
    vision: true
    temperature: 0.0

ollama:
  num_ctx: 8192
  max_num_ctx: 16384                          # ceiling when auto-growing
  num_predict: 1024
  repeat_penalty: 1.15
  repeat_last_n: 256
  image_scale_ladder: [0.7, 0.5]
  empty_response_temperature: 0.35
  disable_json_format_for: ["minicpm-v", "moondream", "granite3.2-vision"]

framework:
  max_image_edge: 896               # was 1280 - biggest single token saving
  confidence_threshold: 0.70        # was 0.60 - activates the feedback loop
  top_k_categories: 3
  enable_feedback_loop: true
```

**Model-diversity rule:** the two annotators must come from **different model families**, or their agreement signal is meaningless. The tie-breaker and finaliser must differ from both, but may share a model with each other — they arbitrate rather than vote, and keeping one model resident saves real time on a long run.

## 5 · Verify everything

```bash
python scripts/check_setup.py
```

Checks the knowledge base, both dataset paths, Ollama connectivity, that all nine model roles are pulled, **and that each one is genuinely multimodal**. Fix anything it flags before continuing.

---

## 6 · Your first run (start small)

Do this before committing to a large batch. ~10 minutes, and it proves the whole loop works.

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

Plenty of `TB` early on is healthy — it means your annotators are genuinely independent. If *everything* is `TB`, they're diverging too much to produce a trustworthy reference.

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
**Overall accuracy** — how often the label *set* matched exactly (strict; no partial credit).

---

## 7 · Validate the manifest before you tune anything

**This step is not optional.** Small VLMs over-apply multi-label annotations — they infer intent rather than observe evidence. In one 100-sample run, **10 of 15 inspected PIH labels were hallucinated**, justified with words like *“implies”*, *“may”*, *“likely”*, or by treating an email/username field as personal information.

If you tune the framework against a wrong answer key, you teach it to replicate the hallucination.

### Inspect the disagreements

```bash
python -c "
import json
rows={json.loads(l)['sample_id']: json.loads(l) for l in open('data/outputs/manifest.jsonl')}
preds={json.loads(l)['sample_id']: json.loads(l) for l in open('data/outputs/predictions.jsonl')}
C='Personal Information Harvesting'
miss=[s for s in preds if s in rows and C in rows[s]['labels'] and C not in preds[s]['labels']]
print(f'{len(miss)} {C} false negatives\n')
for s in miss[:15]:
    print(s)
    print('  manifest :', rows[s]['labels'])
    print('  evidence :', rows[s].get('evidence',{}).get(C))
    print('  screenshot:', rows[s]['screenshot_path'])
"
```

Open ~10 of those screenshots and judge for yourself.

* **Evidence is genuine** (address, DOB, national ID, several identity fields) → the framework is under-calling. Tune it.
* **Evidence is speculative** (“implies”, “may request”, bare email field) → **the manifest is wrong.** Fix the annotator prompt and rebuild.

### Sanity check on prevalence

The paper's human ground truth has PIH on **19.8%** of samples. If your manifest is far above that, your annotators are over-applying it.

```bash
python -c "
import json
from collections import Counter
rows=[json.loads(l) for l in open('data/outputs/manifest.jsonl')]
c=Counter(l for r in rows for l in r['labels'])
for k,v in c.most_common(): print(f'{k:<36}{v:>5}  {v/len(rows):>6.1%}')
print(f'{\"single-annotator rows\":<36}'
      f'{sum(1 for r in rows if r[\"agreement\"]==\"single annotator\"):>5}')
"
```

### Hardening the annotators against speculation

If validation shows hallucination, tighten the labelling rules at the end of `ANNOTATOR_PROMPT` in `src/phishintentionllm/annotation/annotators.py`:

```python
Labelling rules:
- List between 1 and 3 intentions, ordered by confidence.
- Every intention needs at least one concrete element you can SEE in the image.
  Never label on inference. If your evidence contains the words "implies",
  "may", "likely", "suggests", "could" or "probably", you do NOT have evidence
  - drop that category.

- Credential Theft: a password, PIN, OTP or security-answer field is visible.

- Personal Information Harvesting requires TWO OR MORE visible input fields
  collecting identity data beyond the account identifier, OR one strong
  identifier (national ID, SSN, Aadhaar, passport, driving licence, tax ID,
  full postal address, date of birth, or an ID-document upload).
  * An email address or username is an ACCOUNT IDENTIFIER, never PII.
    "Email or mobile number" as a single sign-in field is Credential Theft ONLY.
  * A phone number alone is NOT sufficient.
  * Do NOT label PIH because a page says "verify your identity", shows a
    Terms/Privacy/Contact link, or because more fields might appear later.
  * A plain email + password login form is Credential Theft ONLY.

- Financial Fraud: card number, CVV, expiry, an amount to pay, or transfer /
  wallet details are visible.

- Malware Distribution: a download, install, update or run action is visible.

- If the page looks legitimate, set is_phishing false and return an empty list.
```

Mirror the same constraints into `counter_indicators` under `"Personal Information Harvesting"` in `data/knowledge_base/specialist_kb.json`, and add the anti-speculation rule to `TIEBREAK_PROMPT` and `FINALIZER_PROMPT` — otherwise the finaliser ratifies the claims instead of rejecting them.

Then rebuild:

```bash
cp data/outputs/manifest.jsonl data/outputs/manifest_v1_backup.jsonl
python scripts/build_manifest.py --n 100 --overwrite
python scripts/run_evaluation.py --errors   # same predictions, corrected reference
```

Keep the backup — the before/after distribution shift is a reportable finding.

### Repair single-annotator rows

If an annotator ever crashes, that row falls back to one opinion with no cross-check, yet still counts as reference truth:

```bash
python scripts/build_manifest.py --n 100 --overwrite
```

---

## 8 · Scaling up

```bash
# Stage A · reference set (plan ~70s per sample)
python scripts/build_manifest.py --n 400 --csv data/outputs/manifest.csv

# Stage B · framework  (~120s per sample: 5-6 VLM calls each)
python scripts/run_predictions.py --n 400 --scope manifest

# Stage B · single-agent baseline for the comparison in your report
python scripts/run_predictions.py --n 400 --scope manifest --mode single

# Evaluate with a per-sample error breakdown
python scripts/run_evaluation.py --errors
```

Both stages skip completed work, so run in resumable chunks and interrupt freely:

```bash
python scripts/build_manifest.py  --n 200
python scripts/run_predictions.py --n 200 --scope manifest --skip-done
```

For long runs, detach the process:

```bash
nohup python scripts/build_manifest.py --n 400 > manifest.log 2>&1 &
tail -f manifest.log
```

### Budget realistically

With 3,349 phishing samples (500 Putra + 2,849 Phish-IRIS), a full pass is **~65 h for the manifest and ~110 h for predictions** — 5+ days. A stratified **300–400 sample** subset gives statistically defensible numbers and finishes overnight. Selection is stratified across both sources by default.

---

## 9 · The UI

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

The agent flow animates live, grouped by stage:

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

> The **Analyse** page takes an uploaded screenshot only. For dataset-scale work use **Framework Predictions**.

---

## 10 · Low-VRAM profile

Keep every model multimodal, and keep the two annotators in different families:

```yaml
models:
  annotator_a:         {model: "moondream",  vision: true, temperature: 0.1}
  annotator_b:         {model: "gemma3:4b",  vision: true, temperature: 0.2}
  manifest_tiebreaker: {model: "gemma3:4b",  vision: true, temperature: 0.0}
  manifest_finalizer:  {model: "gemma3:4b",  vision: true, temperature: 0.0}
  vision:              {model: "moondream",  vision: true, temperature: 0.1}
  context:             {model: "gemma3:4b",  vision: true, temperature: 0.0}
  classifier:          {model: "gemma3:4b",  vision: true, temperature: 0.0}
  specialist:          {model: "gemma3:4b",  vision: true, temperature: 0.0}
  validator:           {model: "gemma3:4b",  vision: true, temperature: 0.0}

ollama:
  num_ctx: 4096
  disable_json_format_for: ["moondream"]
framework:
  max_image_edge: 768
```

---

## 11 · Optional: human ground truth

The manifest is machine-generated. For a stronger claim, add human labels using the paper's protocol — two engineers label independently, a reviewer signs off:

1. **✍️ Manual Annotation** → select your identity → label → repeat as the second engineer → then as `reviewer`.
2. Export and evaluate:

```bash
python scripts/build_ground_truth.py
python scripts/run_evaluation.py --reference ground_truth
```

The reviewer can also verify machine-generated manifest rows in place (queue: *⚠️ Manifest flagged for review*), then score against only those:

```bash
python scripts/run_evaluation.py --verified-only
```

---

## 12 · Troubleshooting

The Ollama client recovers from model failures by **changing the request** before retrying — an identical retry can never fix a decoding failure. Three ladders handle the common cases automatically.

| Symptom | What happens automatically | Manual fix if it persists |
|---|---|---|
| `token repeat limit reached` | Drops the JSON grammar → retries shorter and warmer | Add the model to `ollama.disable_json_format_for`, or set `models.<role>.json_format: false` |
| `exceeds the available context size` | Raises `num_ctx` → down-scales the image → trims the prompt | Lower `framework.max_image_edge` to `896`; raise `models.<role>.num_ctx` |
| `Empty response from model` | Drops the grammar + adds a JSON primer → warms temperature → shrinks image and prompt | Raise `models.<role>.temperature` to ~0.3; set `json_format: false`; swap the model |
| `(single annotator)` in the log | — | An annotator crashed; re-run with `--overwrite` |
| `VisionCapabilityError` at startup | — | A role points at a text-only model — swap it in `config.yaml` |
| Everything labelled Credential Theft | — | Raise `framework.confidence_threshold`; strengthen `I_c` indicators in `specialist_kb.json` |
| PIH labelled on speculation | — | Harden the annotator prompt — see [§7](#7--validate-the-manifest-before-you-tune-anything) |
| Annotators always agree (IAA = 100%) | — | Your annotator models are too similar — use different families |
| Tie-breaker never fires | — | Lower `manifest.agreement_margin` |
| `feedback_loops: 0` across a whole run | — | The validator is overconfident — raise `framework.confidence_threshold` to 0.70 |
| Sector matrix has duplicates (`telecoms` *and* `telecommunications`) | — | Known issue: the validator's free-text sector overwrites Layer 1's normalised value. Route `vmeta["sector"]` through `VisionAnalysisAgent._normalise_sector()` in `orchestrator.py` |
| Too slow | — | Lower `max_image_edge`; smaller models; `enable_feedback_loop: false` |

### Test without Ollama

```bash
python tests/test_pipeline.py          # metrics, consensus, KB, manifest store
python tests/test_end_to_end_mock.py   # both stages + a VLM-enforcement audit
```

The second prints an audit proving every agent call carried the screenshot.

---

## 13 · Benchmarking against the paper

The paper evaluates on a **2,063-sample human-labelled** ground truth. A local run on a machine-generated manifest is not like-for-like — state that explicitly.

**Paper results (Table 2):**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| GPT-4o | 0.7895 | 0.8544 | 0.8207 | 0.8915 |
| Gemini 2.0 | 0.7843 | 0.8976 | 0.8371 | 0.8987 |
| GPT-4o-mini | 0.6149 | 0.9744 | 0.7540 | 0.8149 |
| Qwen2.5-VL-72B | 0.4520 | 0.9428 | 0.6111 | 0.6518 |

**Credential-theft benchmark (Table 5)** — the fairest single comparison, since the paper reports this class on its own:

| | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| PhishIntention (prior work) | 0.8206 | 0.7471 | 0.7821 | 0.6578 |
| PhishIntentionLLM (GPT-4o) | 0.8545 | 0.9946 | 0.9193 | 0.8602 |

A local open-weight stack typically lands **higher on precision** and **lower on recall** than the commercial models — the opposite error profile. Report per-class metrics as well as micro-averages; the aggregate hides where the loss actually is.

Useful commands for the write-up:

```bash
python scripts/run_evaluation.py --errors           # per-sample exact/partial/miss
python scripts/run_evaluation.py --compare \
    data/outputs/runs/run_<framework_id>.json \
    data/outputs/runs/run_<baseline_id>.json        # Table 4 analogue
```

The **Evaluation** page also reports *reference quality* — inter-annotator agreement, tie-break rate and review backlog — which tells a reviewer how much to trust the manifest your numbers rest on.

---

## 14 · Architecture reference

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

| Stage | Role | Recommended model |
|---|---|---|
| A | `annotator_a` | `minicpm-v:8b` |
| A | `annotator_b` | `gemma3:4b` |
| A | `manifest_tiebreaker` | `mistral-small3.2:24b` |
| A | `manifest_finalizer` | `mistral-small3.2:24b` |
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

Plus set-level agreement, the credential-theft benchmark, co-occurrence and sector × intention analyses, and a reference-quality panel.

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
│   ├── llm/                     # vision-enforcing Ollama client + recovery ladders
│   └── utils/                   # image encoding, tolerant JSON parsing, logging
├── data/
│   ├── knowledge_base/          # K_B and K_c
│   └── outputs/                 # manifest.jsonl · predictions.jsonl · runs/
├── scripts/                     # check_setup, build_manifest, run_predictions, …
└── tests/                       # offline unit tests + mocked end-to-end test
```

See `docs/ARCHITECTURE.md` for the full paper-to-code mapping.

---

## 15 · Command reference

```bash
# setup
python scripts/check_setup.py

# Stage A · manifest
python scripts/build_manifest.py --n 400
python scripts/build_manifest.py --n 400 --overwrite          # re-annotate
python scripts/build_manifest.py --n 50 --sources putra       # one source only
python scripts/build_manifest.py --n 50 --csv out.csv         # + CSV export

# Stage B · predictions
python scripts/run_predictions.py --n 400 --scope manifest
python scripts/run_predictions.py --n 400 --scope manifest --mode single
python scripts/run_predictions.py --n 400 --scope random --skip-done

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
