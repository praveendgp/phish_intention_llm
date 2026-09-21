# Architecture & Paper Mapping

How the implementation maps to the base paper, and where the project brief
deliberately diverges from it.

---

## 1. The two-stage separation

The brief assigns the annotators a distinct job from the framework, so the code
keeps them in separate packages with separate model groups:

| | Stage A — MANIFEST | Stage B — FRAMEWORK |
|---|---|---|
| Package | `annotation/` | `agents/` + `pipeline/` |
| Config roles | `annotator_a`, `annotator_b`, `manifest_tiebreaker`, `manifest_finalizer` | `vision`, `context`, `classifier`, `specialist`, `validator` |
| Output | `data/outputs/manifest.jsonl` | `data/outputs/predictions.jsonl` |
| Purpose | produce **reference labels** | the **system under test** |
| Sees the other's output? | no | **no** — this is what keeps evaluation honest |

`Evaluator` joins the two on `sample_id` and computes the metric suite.

### Why the framework must not see the manifest
If the framework's classification layer consumed annotator votes, the evaluation
would measure agreement with its own input rather than independent performance.
`ClassificationAgent` therefore does its own multi-label scoring from the
screenshot, exactly as `CLASSIFICATIONAGENT(E_enriched, features)` does in the
paper.

---

## 2. Algorithm correspondence

`pipeline/orchestrator.py` implements Fig. 3 of the paper:

| Paper (Fig. 3) | Implementation |
|---|---|
| `E ← VISIONANALYSISAGENT(I)` | `agents/vision_agent.py` |
| `patterns ← RETRIEVEPATTERNS(K_B)` | `rag/knowledge_base.py :: retrieve_patterns` |
| `E_enriched ← CONTEXTENRICHMENTAGENT(E, patterns)` | `agents/context_agent.py` |
| `features ← RETRIEVECATEGORYFEATURES(K_C)` | `knowledge_base.retrieve_category_features` |
| `C, S ← CLASSIFICATIONAGENT(E_enriched, features)` | `agents/classification_agent.py` |
| `P ← {c ∈ C | Top-k(S, k=3)}` | `ClassificationAgent._select_candidates` |
| `knowledge ← RETRIEVESPECIALISTKNOWLEDGE(K_C, c)` | `knowledge_base.retrieve_specialist_knowledge` |
| `A_c ← SPECIALISTAGENT(E_enriched, knowledge, c)` | `agents/specialist_agents.py` |
| `R, conf ← VALIDATIONAGENT(...)` | `agents/validation_agent.py` |
| `if conf < τ` → widen to `categories \ P`, re-validate | `orchestrator._run_specialists(remaining, …)` |
| `T ← {(t, evidence, conf) | conf ≥ τ}` | `orchestrator._format_results` |
| `if |T| = 0 → T ← {argmax conf}` | same function, fallback branch |

τ is `framework.confidence_threshold` (default `0.60`).

---

## 3. VLM enforcement — how it is guaranteed

The brief requires every agent to analyse the image with a VLM. Four
independent checks make that non-negotiable:

1. **Agent construction** — `VisionAgent.__init__` raises
   `VisionCapabilityError` if its role's model declares `vision: false`
   (while `framework.vision_for_all_agents` is true).
2. **Call boundary** — `VisionAgent.ask_json(prompt, image_b64, …)` takes the
   image as a **required positional argument** and raises if it is empty. There
   is no code path in which an agent reasons without the screenshot.
3. **Transport** — `OllamaClient.generate()` refuses to attach images to a
   model marked non-vision.
4. **Capability probe** — `OllamaClient.supports_vision()` queries `/api/show`
   for CLIP/vision families; surfaced in `check_setup.py` and the Settings page.

**Observability.** Each `AgentStep` carries `vision: true`, the UI renders a
`VLM · sees image` badge, `AnalysisResult.vision_calls` counts the calls, and
`tests/test_end_to_end_mock.py` asserts `client.text_calls == 0` across a full
run of both stages.

**Efficiency.** The screenshot is encoded to base64 PNG **once per sample**
(`utils/image.load_image_b64`, down-scaled to `max_image_edge`) and that single
payload is handed to all nine agents.

---

## 4. Manifest construction

`annotation/manifest_builder.py`:

```
1. Annotator A  (VLM, no knowledge of B)      -> AnnotatorVote
2. Annotator B  (VLM, no knowledge of A)      -> AnnotatorVote
3. analyse_agreement()                        -> full / partial / conflict
4. Tie-Breaker  (VLM)  only on disputes       -> keep/drop per category
5. Finaliser    (VLM)                         -> authoritative label set
6. ManifestRecord appended to manifest.jsonl
```

* **Escalation rule.** A category claimed by one annotator only goes to the
  tie-breaker unless that annotator's confidence ≥ `1 − agreement_margin`
  (default 0.85). Agreed categories are never re-litigated, keeping cost down.
* **Provenance.** Every row stores both votes, the agreement status, the
  tie-break decisions, the full agent trace and the finaliser's justification —
  so any label can be audited later.
* **Review flags.** Rows are flagged via `manifest.flag_for_review_when`
  (conflict, tie-break applied, low confidence) and surface in the Manual
  Annotation queue.
* **Graceful degradation.** If the finaliser fails, `_fallback_payload`
  computes a deterministic consensus (agreed set + tie-break keeps) and marks
  the row for review rather than dropping the sample.
* **Idempotence.** `build(..., overwrite=False)` skips samples already present;
  a later line for the same `sample_id` supersedes the earlier one.

---

## 5. Model selection constraints

| Constraint | How it is satisfied |
|---|---|
| Open-source models via Ollama | All nine roles are local Ollama models; the client speaks plain HTTP |
| Do not use the paper's annotator models | GPT-4o, GPT-4o-mini, Gemini-2.0-Flash, Qwen2.5-VL-72B appear nowhere; asserted in `test_every_model_is_a_vlm` |
| No LLaMA models | No `llama*` or LLaMA-derived checkpoint is configured; asserted in the same test |
| 2 annotators + 1 finaliser + 1 tie-breaker | `annotator_a`, `annotator_b`, `manifest_finalizer`, `manifest_tiebreaker` |
| Annotators create the manifest | `ManifestBuilder` writes `manifest.jsonl`; nothing else does |
| All agents use a VLM | See §3 |

---

## 6. Manual annotation retained

`annotation/manual.py` keeps the paper's human protocol as a first-class
feature — two independent labellers, a reviewer with final say, multi-intention
labels, agreement detection and consensus resolution
(`reviewer → unanimous → majority → union`).

It additionally serves as the **human verification path for the manifest**:
`verify_manifest_row()` lets the reviewer correct an annotator-generated row,
sets `human_verified: true`, clears `needs_review` and records the decision in
the manual log. The Evaluation page can then be restricted to human-verified
rows only.

---

## 7. Dataset normalisation

| | Putra | Phish-IRIS |
|---|---|---|
| Layout | `<label>/<record-id>/screenshots/*.png` + `assets/` | `<split>/<brand>/*.png` |
| Label | folder (`phishing` / `not-phishing`) | always phishing |
| Brand | unknown | folder name |
| Extras | URL/metadata harvested from `assets/` | brand → sector prior |
| Multiple shots | best one chosen, rest kept as metadata | one image per sample |

`datasets/registry.py` merges them, supports stratified sampling across sources
and degrades gracefully when one source is absent.

---

## 8. Robustness engineering

Small open-weight VLMs are less reliable than the paper's commercial models, so:

* **Tolerant JSON recovery** — fence stripping, balanced-brace extraction,
  trailing-comma repair, `ast.literal_eval` fallback.
* **Score coercion** — `85%`, `8.5/10`, `85`, `0.85` all normalise to `0.85`.
* **Category coercion** — `"credentials theft"`, `"CT"`, `"credential_theft"`
  all map to the canonical label.
* **Per-agent isolation** — a failing agent marks its own node `error` without
  collapsing the run; batch runners continue.
* **Retries with backoff** on the Ollama transport.
* **Guaranteed non-empty output** in both stages (`|T| ≥ 1`).

---

## 9. Evaluation

`evaluation/metrics.py` implements Eq. 3–10 exactly, including the partial-match
thresholds `t_k = 1` for k ∈ {1,2} and `t_k = 2` for k = 3, plus `evaluate_binary`
for the credential-theft benchmark, `agreement_rate` for set-level comparison
and `co_occurrence` / `sector_matrix` for the large-scale profiling analysis.

`evaluation/evaluator.py` adds:

* `evaluate_run(reference="manifest" | "ground_truth", verified_only=…)`
* `per_sample()` — row-by-row exact / partial / miss error analysis
* `compare_runs()` — the Table 4 framework-vs-baseline experiment
* `manifest_quality()` — inter-annotator agreement, tie-break rate, review
  backlog, so the reliability of the reference is reported alongside the score

---

## 10. Extension points

| Goal | Where |
|---|---|
| Add a fifth intention category | `schemas.Intention`, `config.categories`, a new block in `specialist_kb.json` — the specialist pool picks it up automatically |
| Add a third annotator | Add `models.annotator_c`, instantiate in `ManifestBuilder`, extend `analyse_agreement` to majority voting |
| Swap the retriever | Set `framework.embedding_model`, or add a class with `.search()` to `rag/retriever.py` |
| Add a dataset source | Subclass `BaseDatasetLoader`, register it in `datasets/registry.LOADERS` |
| Use URL/HTML features | Extend `VisualElements` and the Layer 2 prompt; Putra `assets/` metadata is already parsed |
