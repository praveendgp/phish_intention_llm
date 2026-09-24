"""🎣 Intent Scanner - analyse a single uploaded screenshot."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, flow_legend, footer, get_config,
                           get_framework, get_store, hero, render_agent_flow,
                           render_intentions, render_label_chips,
                           render_specialists, render_verdict, sidebar_status,
                           style_page)

style_page()
cfg = get_config()
store = get_store()

hero("PhishIntentionLLM",
     "Uncovering <b>why</b> a phishing site exists — not just that it is one. "
     "Five cooperating vision-language layers, a dual-layer knowledge base and "
     "evidence-backed verdicts, all running locally.",
     tags=["Multi-Agent RAG", "Every agent is a VLM", "4 intention classes",
           "Local Ollama models", "MTech Project"])
sidebar_status()

# ---------------------------------------------------------------------------
# 1 · Upload
# ---------------------------------------------------------------------------
st.markdown("### 1 · Upload a screenshot")

image_path = sample_id = None
brand_hint = None
source = "upload"

uploaded = st.file_uploader(
    "Website screenshot",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
    help="Drop in any phishing or legitimate website screenshot.",
)

if uploaded is not None:
    upload_dir = Path(cfg.resolve("storage.results_dir")) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{int(time.time())}_{uploaded.name}"
    target.write_bytes(uploaded.getbuffer())
    image_path = str(target)
    sample_id = f"upload::{uploaded.name}"

# ---------------------------------------------------------------------------
# 2 · Run
# ---------------------------------------------------------------------------
st.markdown("### 2 · Run the agent pipeline")
left, right = st.columns([1, 1.45])

with left:
    if image_path:
        st.image(image_path, caption=Path(image_path).name, **FULL_WIDTH)
    else:
        st.info("Upload a screenshot to begin.")

    mode = st.radio("Pipeline", ["PhishIntentionLLM (5-layer multi-agent RAG)",
                                 "Single-agent baseline"], index=0)
    save_result = st.checkbox("Save this prediction for evaluation", value=True)
    launch = st.button("🚀 Analyse intentions", type="primary",
                       disabled=not image_path, **FULL_WIDTH)

with right:
    flow_placeholder = st.empty()
    flow_placeholder.markdown(render_agent_flow([]), unsafe_allow_html=True)
    flow_legend("framework")

if launch and image_path:
    steps: list = []

    def on_step(step):
        if step not in steps:
            steps.append(step)
        flow_placeholder.markdown(render_agent_flow(steps, compact=True),
                                  unsafe_allow_html=True)

    with st.spinner("Agents are examining the screenshot…"):
        if mode.startswith("PhishIntentionLLM"):
            result = get_framework().analyse_screenshot(
                image_path, on_step=on_step, brand_hint=brand_hint,
                sample_id=sample_id, source=source)
        else:
            from phishintentionllm.pipeline import SingleAgentBaseline
            engine = SingleAgentBaseline(cfg, get_framework().client)
            result = engine.analyse_screenshot(image_path, on_step=on_step,
                                               sample_id=sample_id, source=source)

    flow_placeholder.markdown(render_agent_flow(result.steps), unsafe_allow_html=True)
    st.session_state["last_result"] = result
    if save_result and not result.error:
        store.save_prediction(result)

# ---------------------------------------------------------------------------
# 3 · Results
# ---------------------------------------------------------------------------
result = st.session_state.get("last_result")
if result is not None:
    st.markdown("### 3 · Assessment")
    if result.error:
        st.error(f"The pipeline reported an error: {result.error}")

    render_verdict(result)

    col_left, col_right = st.columns([1.35, 1])
    with col_left:
        st.markdown("#### 🎯 Identified intentions")
        render_intentions(result.intentions)
        if result.recommended_action:
            st.success(f"**Recommended action:** {result.recommended_action}")

    with col_right:
        st.markdown("#### 📊 Score for every category")
        scores = {s.category: s.score for s in result.category_scores}
        for intention in result.intentions:
            scores[intention.category] = max(scores.get(intention.category, 0),
                                             intention.confidence)
        bar_chart(scores, height=210)

        st.markdown("#### 🔬 Expert panel")
        render_specialists(result)

    reference = store.manifest_map().get(result.sample_id)
    if reference is not None:
        st.markdown("#### 📋 Versus the manifest reference")
        ref_col, pred_col = st.columns(2)
        with ref_col:
            render_label_chips(reference, "Manifest (annotators):")
        with pred_col:
            render_label_chips(result.labels, "Framework prediction:")
        if set(reference) == set(result.labels):
            st.success("Exact match with the annotator-generated reference.")
        elif set(reference) & set(result.labels):
            st.warning("Partial match.")
        else:
            st.error("No overlap with the reference labels.")

    with st.expander("🔍 Full agent trace"):
        st.markdown(render_agent_flow(result.steps), unsafe_allow_html=True)

    if cfg.get("ui.show_raw_json", True):
        with st.expander("🧾 Raw JSON result"):
            st.json(result.to_dict())
        st.download_button("⬇️ Download result (JSON)",
                           data=json.dumps(result.to_dict(), indent=2,
                                           ensure_ascii=False),
                           file_name=f"phishintention_{result.run_id}.json",
                           mime="application/json")

footer()
