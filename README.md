# 🎣 PhishIntentionLLM

**Uncovering Phishing Website Intentions through Multi-Agent Retrieval-Augmented Generation**

A faithful re-implementation of *PhishIntentionLLM* (Li, Manickam, Chong & Karuppayah — ICDF2C 2025), built for an MTech project and adapted to run entirely on **local open-source vision-language models served by Ollama**.

The framework answers a question ordinary phishing detectors ignore: not *“is this page phishing?”* but ***“what does the attacker actually want?”*** — across four intentions:

| | Intention | What the attacker is after |
|---|---|---|
| 🔑 | **Credential Theft** | usernames, passwords, OTPs |
| 💳 | **Financial Fraud** | card data, transfers, payments |
| 🦠 | **Malware Distribution** | getting a payload executed |
| 🪪 | **Personal Information Harvesting** | identity data, KYC documents |

---

## 🧭 Two stages, one clean separation

This is the core design of the project:

```
                    ┌──────────────────── STAGE A · MANIFEST ─────────────────────┐
  screenshot ──────►│  Annotator A (VLM) ─┐                                       │
                    │                      ├─► tie-break ─► finaliser ─► manifest │────┐
                    │  Annotator B (VLM) ─┘   (on dispute)              .jsonl    │    │
                    └─────────────────────────────────────────────────────────────┘    │
                                                                                       │   reference
                    ┌──────────────── STAGE B · FRAMEWORK (5 layers) ───────────────┐  │ labels
  screenshot ──────►│ 👁️ Vision → 🧠 Context → 🏷️ Classify → 🔬 Experts → ✅ Validate │──│
                    └───────────────────────────────────────────────────────────────┘  │
                                                                                       ▼
                                                                                📊 EVALUATION
```

* **Annotators build the manifest.** Two independent VLMs label every screenshot; a tie-breaker rules on genuine disputes; a finaliser signs off one manifest row per sample. `manifest.jsonl` is the **reference label set**.
* **The framework predicts.** The five paper layers analyse the same screenshot and emit `predictions.jsonl`. It never reads the manifest, so the evaluation stays honest.
* **Evaluation scores one against the other** using the paper's metric suite.

## 👁️ Every agent is a VLM

**All nine agents receive the screenshot itself**, not a text transcript of it. This is enforced in code, not just by convention:

* `VisionAgent.ask_json()` **requires** an `image_b64` argument and raises `VisionCapabilityError` without one.
* Agent construction fails fast if its configured model declares `vision: false`.
* `OllamaClient.generate()` refuses to send images to a non-vision model.
* The UI stamps every flow node with a **`VLM · sees image`** badge, and each result reports how many vision calls it made.
* `scripts/check_setup.py` probes `/api/show` and flags any model that is not multimodal.

Layer 2 even folds elements it spots into the record under `missed_elements` when the perception layer under-reports them — only possible because it looks at the image itself.

---

## 🤖 Model line-up (constraints respected)

> None of the models evaluated in the paper (GPT-4o, GPT-4o-mini, Gemini-2.0-Flash, Qwen2.5-VL-72B) are used as annotators, and **no LLaMA-family model** appears anywhere.

**Stage A — manifest (annotators)**

| Role | Default model | Job |
|---|---|---|
| `annotator_a` | `minicpm-v:8b` | Independent label #1 |
| `annotator_b` | `granite3.2-vision:2b` | Independent label #2 |
| `manifest_tiebreaker` | `mistral-small3.2:24b` | Rules on disputed categories only |
| `manifest_finalizer` | `gemma3:12b` | Signs off the manifest row |

**Stage B — framework (the five paper layers)**

| Role | Default model | Layer |
|---|---|---|
| `vision` | `minicpm-v:8b` | 1 · Perception |
| `context` | `gemma3:12b` | 2 · Context enrichment |
| `classifier` | `gemma3:12b` | 3 · Multi-label classification |
| `specialist` | `gemma3:12b` | 4 · Four expert agents |
| `validator` | `mistral-small3.2:24b` | 5 · Validation & synthesis |

```bash
ollama pull minicpm-v:8b
ollama pull granite3.2-vision:2b
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
```

*Lighter machine?* Use `moondream` + `granite3.2-vision:2b` as annotators and `gemma3:4b` for everything else — all still multimodal.

---

## 🚀 Quick start

```bash
# 1 · install
pip install -r requirements.txt

# 2 · start Ollama (separate terminal) and pull the four models
ollama serve

# 3 · drop the datasets in place  (see data/raw/README.md)
#     data/raw/putra/phishing/<record-id>/screenshots/*.png
#     data/raw/phishIris/train/<brand>/*.png

# 4 · verify everything (including VLM capability)
python scripts/check_setup.py

# 5 · launch the console
./run_ui.sh              # Windows: run_ui.bat
```

Then open <http://localhost:8501>.

---

## 🖥️ The console

| Page | What you do there |
|---|---|
| **🎣 Analyse** | Run one screenshot through the five layers; watch the agent flow live; compare the verdict against the manifest |
| **📚 Dataset Explorer** | Browse both corpora, filter by brand, see which samples already have manifest labels |
| **📋 Manifest Builder** | Run the annotator ensemble; inspect every manifest row, its votes, agreement and tie-break; export CSV/JSONL |
| **🤖 Framework Predictions** | Batch-run the framework (or baseline) — optionally restricted to manifest-covered samples |
| **📊 Evaluation** | Micro metrics, Acc_comp, per-class table, error analysis, reference quality, framework vs baseline |
| **✍️ Manual Annotation** | Two engineers + reviewer protocol — and reviewer sign-off on flagged manifest rows |
| **🧠 Knowledge Base** | Inspect K_B and K_c; test retrieval queries interactively |
| **⚙️ Settings** | Ollama health, both model groups, multimodal capability audit |

The agent flow renders as a live, stage-grouped chain:

```
MANIFEST STAGE
🗳️  Annotator A            [VLM · sees image]   Labelled: CT 0.92          ● Done
🗳️  Annotator B            [VLM · sees image]   Labelled: CT 0.92, PIH 0.55 ● Done
⚖️  Tie-Breaker            [VLM · sees image]   Ruled on 1; kept: none      ● Done
📝  Manifest Finaliser     [VLM · sees image]   Labels: CT (quality: high)  ● Done

FRAMEWORK STAGE
👁️  Vision Analysis Agent  [VLM · sees image]   Saw 2 form fields, password ● Done
🧠  Context Enrichment     [VLM · sees image]   Tagged 1, +1 missed element ● Done
🏷️  Classification Agent   [VLM · sees image]   Nominated CT 0.93           ● Done
🔬  Credential Theft Expert[VLM · sees image]   CONFIRMED CT @ 0.93         ● Done
🔬  Financial Fraud Expert                      Not nominated               ◌ Skipped
✅  Validation & Synthesis [VLM · sees image]   Final: CT 0.93              ● Done
```

---

## 🧩 Architecture

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

**Knowledge architecture (Definition 1)**

* `K_B = {P_c, P_v, P_t, P_u}` — common patterns, visual deception, text manipulation, URL red flags
* `K_c = {F_c, D_c}` with `D_c = {T_c, M_c, I_c}` — per-category features, targets, techniques, indicators

Both live in `data/knowledge_base/*.json`. Retrieval uses a dependency-free TF-IDF index by default; set `framework.embedding_model` for sentence-transformers.

---

## 📁 Project layout

```
PhishIntentionLLM/
├── config.yaml                  # models (2 groups), τ, datasets, storage
├── run_ui.sh / run_ui.bat
├── app/
│   ├── streamlit_app.py         # main "Analyse" console
│   ├── ui_components.py         # theme, stage-aware flow renderer, result cards
│   └── pages/                   # 7 additional pages
├── src/phishintentionllm/
│   ├── agents/                  # base (VLM enforcement) + the 5 framework layers
│   ├── annotation/              # annotators.py, manifest_builder.py, manual.py, store.py
│   ├── datasets/                # putra.py, phishiris.py, registry.py
│   ├── rag/                     # knowledge_base.py, retriever.py
│   ├── pipeline/                # orchestrator.py, single_agent.py, batch.py
│   ├── evaluation/              # metrics.py (Eq. 3-10), evaluator.py
│   ├── llm/ollama_client.py     # vision-enforcing HTTP client
│   └── utils/                   # image encoding, tolerant JSON parsing, logging
├── data/
│   ├── knowledge_base/          # K_B and K_c
│   └── outputs/                 # manifest.jsonl · predictions.jsonl · runs/
├── scripts/                     # check_setup, build_manifest, run_predictions, ...
└── tests/                       # offline unit tests + mocked end-to-end test
```

---

## ⌨️ Command line

```bash
# health check (includes multimodal capability probe)
python scripts/check_setup.py

# STAGE A - annotators build the reference manifest
python scripts/build_manifest.py --n 50 --csv data/outputs/manifest.csv

# STAGE B - framework predicts on manifest-covered samples
python scripts/run_predictions.py --n 50 --scope manifest

# single-agent baseline for comparison
python scripts/run_predictions.py --n 50 --scope manifest --mode single

# evaluate predictions against the manifest
python scripts/run_evaluation.py --errors

# optional: human-reviewed ground truth instead of the manifest
python scripts/build_ground_truth.py
python scripts/run_evaluation.py --reference ground_truth
```

---

## 🧪 Tests

```bash
python tests/test_pipeline.py          # metrics, consensus, KB, manifest store, VLM config
python tests/test_end_to_end_mock.py   # both stages + a VLM-enforcement audit
```

Neither needs Ollama. The end-to-end test asserts that **every agent call carried an image** and that zero text-only calls occurred.

---

## 📐 Metrics implemented

| Metric | Equation in paper |
|---|---|
| Overall accuracy (exact set match) | Eq. 3 |
| Precision / Recall / F1 (micro) | Eq. 4–6 |
| Accuracy by complexity `Acc_comp(k)`, `t_k = 1,1,2` | Eq. 7–8 |
| Per-class P / R / F1 / Accuracy | Eq. 9–10 |

Plus set-level agreement (exact / partial / Jaccard), the credential-theft benchmark, co-occurrence and sector × intention analyses, and a **reference-quality panel** (inter-annotator agreement, tie-break rate, review backlog) so you can judge how much to trust the manifest itself.

---

## 🔧 Tuning notes

| Symptom | Fix |
|---|---|
| `VisionCapabilityError` at startup | A role points at a text-only model — swap it in `config.yaml` |
| Manifest labels look noisy | Raise `manifest.min_confidence`; review flagged rows on the Manual Annotation page |
| Annotators always agree (IAA = 100%) | Your two annotator models are too similar — pick more diverse architectures |
| Tie-breaker never fires | Lower `manifest.agreement_margin` (a solo claim escalates when confidence < 1 − margin) |
| Everything labelled Credential Theft | Raise `framework.confidence_threshold`; strengthen `I_c` indicators for other categories |
| Pipeline too slow | Smaller models, lower `max_image_edge`, or `enable_feedback_loop: false` |
| Out of VRAM | Shrink `ollama.num_ctx`, use 2–4B VLMs, reduce `ollama.keep_alive` |

---

## 📚 Reference

Li, W., Manickam, S., Chong, Y.-W., & Karuppayah, S. (2025). *PhishIntentionLLM: Uncovering Phishing Website Intentions through Multi-Agent Retrieval-Augmented Generation.* ICDF2C 2025. arXiv:2507.15419

Datasets: Putra phishing website dataset (Zenodo 8041387) · Phish-IRIS .

> Research and educational use only. Handle phishing screenshots in an isolated environment.
