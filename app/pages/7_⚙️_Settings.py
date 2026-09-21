"""Runtime settings: Ollama health, the two model groups, and VLM compliance."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import FULL_WIDTH, footer, get_config, get_framework, hero, setup_page

setup_page("Settings", "⚙️")
cfg = get_config()

hero("Settings & Runtime",
     "Two model groups, one rule: every agent analyses the screenshot with a "
     "vision-language model.",
     tags=["Annotators → manifest", "Framework → predictions",
           "No paper models", "No LLaMA models"])

# ---------------------------------------------------------------------------
st.markdown("### 🔌 Ollama connection")
report = get_framework().preflight()

if report.get("alive"):
    st.success(f"Connected to Ollama at `{report['host']}`")
else:
    st.error(f"Cannot reach Ollama at `{report['host']}`")
    st.code("ollama serve", language="bash")

# ---------------------------------------------------------------------------
ROLE_HELP = {
    "annotator_a": "Independent annotator #1 — writes a vote into the manifest",
    "annotator_b": "Independent annotator #2 — second, independent vote",
    "manifest_tiebreaker": "Rules on categories the annotators disputed",
    "manifest_finalizer": "Signs off the manifest row for each sample",
    "vision": "Layer 1 — perception: OCR, form fields, buttons, branding",
    "context": "Layer 2 — enriches observations with K_B threat patterns",
    "classifier": "Layer 3 — scores all four categories, nominates the top-k",
    "specialist": "Layer 4 — the four category expert agents",
    "validator": "Layer 5 — validation, synthesis and the final verdict",
}


def render_group(stage: str, title: str, caption: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(caption)
    rows = []
    for role, info in report.get("models", {}).items():
        if info["stage"] != stage:
            continue
        if info["detected_vision"] is True:
            vision = "✅ multimodal"
        elif info["detected_vision"] is False:
            vision = "❌ NOT multimodal"
        else:
            vision = "➖ unknown"
        rows.append({"Role": role, "Purpose": ROLE_HELP.get(role, ""),
                     "Model": info["model"],
                     "Pulled": "✅" if info["ready"] else "⬇️ missing",
                     "Vision capability": vision})
    st.dataframe(rows, hide_index=True, **FULL_WIDTH)


st.markdown("### 🧩 Model roles")
render_group("manifest", "A · Manifest stage (annotators)",
             "These four models produce `manifest.jsonl` — the reference labels "
             "used to measure the framework's performance.")
render_group("framework", "B · Framework stage (the five paper layers)",
             "The system under test. It never reads the manifest, so its "
             "predictions stay independent of the reference.")

if report.get("missing"):
    st.warning("Pull the missing models before running anything:")
    st.code("\n".join(f"ollama pull {m}" for m in report["missing"]), language="bash")

if report.get("non_vision"):
    st.error("These models are not multimodal, but every agent must analyse the "
             "screenshot itself: " + ", ".join(report["non_vision"])
             + "\n\nReplace them in `config.yaml` with vision-capable models.")
else:
    st.success("All configured models are vision-capable — every agent analyses "
               "the image directly.")

with st.expander("Models installed on this machine"):
    st.write(report.get("available", []))

st.info("**Constraint compliance** — none of the models evaluated in the base "
        "paper (GPT-4o, GPT-4o-mini, Gemini-2.0-Flash, Qwen2.5-VL-72B) are used "
        "as annotators, and no LLaMA-family model appears anywhere in the project.")

# ---------------------------------------------------------------------------
st.markdown("### 🎛️ Framework parameters")
params = cfg.get("framework", {})
cols = st.columns(4)
cols[0].metric("τ confidence threshold", params.get("confidence_threshold"))
cols[1].metric("Top-k categories", params.get("top_k_categories"))
cols[2].metric("Retrieval top-k", params.get("retrieval_top_k"))
cols[3].metric("Max image edge", f"{params.get('max_image_edge')} px")

toggles = st.columns(3)
toggles[0].caption(f"Feedback loop: **{params.get('enable_feedback_loop')}**")
toggles[1].caption(f"RAG enabled: **{params.get('enable_rag')}**")
toggles[2].caption(f"Vision for all agents: **{params.get('vision_for_all_agents')}**")

st.markdown("### 📋 Manifest parameters")
manifest = cfg.get("manifest", {})
m_cols = st.columns(4)
m_cols[0].metric("Agreement margin", manifest.get("agreement_margin"))
m_cols[1].metric("Min confidence", manifest.get("min_confidence"))
m_cols[2].metric("Max intentions", manifest.get("max_intentions"))
m_cols[3].metric("Evaluation reference", cfg.get("evaluation.reference"))
st.caption("Flag for human review when: "
           + ", ".join(manifest.get("flag_for_review_when", [])))

st.markdown("### 🗂️ Storage paths")
st.dataframe([{"Key": key, "Path": str(cfg.resolve(f"storage.{key}"))}
              for key in cfg.get("storage", {})], hide_index=True, **FULL_WIDTH)

st.markdown("### 📄 Current configuration")
with st.expander("config.yaml"):
    st.code(yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True),
            language="yaml")
st.caption(f"Loaded from `{cfg.path}` — edit the file and restart the app to apply.")

footer()
