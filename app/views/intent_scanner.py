"""🎣 Intent Scanner - analyse a screenshot (Tiered Free vs Pro + Live Execution)."""

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
from phishintentionllm.auth import current_plan, render_auth_widget, render_login_dialog_button
from phishintentionllm.schemas import (AnalysisResult, IntentionResult,
                                       SpecialistFinding, CategoryScore, AgentStep)

style_page()
cfg = get_config()
store = get_store()

hero("PhishIntentionLLM",
     "Uncovering <b>why</b> a phishing site exists — not just that it is one. "
     "Five cooperating vision-language layers, a dual-layer knowledge base and "
     "evidence-backed verdicts, all running locally.",
     tags=["Multi-Agent RAG", "Every agent is a VLM", "4 intention classes",
           "Local Ollama models", "MTech Project"])

# 1. MOVED TO TOP OF SIDEBAR: Render login widget FIRST, then hardware/model status below it
render_auth_widget(cfg)
sidebar_status()

# Check active tier
plan = current_plan(cfg)
is_pro = plan.get("can_upload", False) and plan.get("can_execute_pipeline", False)

# ---------------------------------------------------------------------------
# 1 · Input Selection
# ---------------------------------------------------------------------------
st.markdown("### 1 · Input Selection")

image_path = None
sample_id = None
brand_hint = None
source = "upload"

if not is_pro:
    col_msg, col_btn = st.columns([3, 1])
    with col_msg:
        st.info("🔒 **Free Community Tier**: Live screenshot uploads and real-time VLM agent runs "
                "are reserved for **Pro** accounts to manage GPU/compute quotas. "
                "You can inspect verified archive runs below.")
    with col_btn:
        # Prominent inline button right in the main UI
        render_login_dialog_button(cfg)

    # Locate pre-computed evaluations
    eval_candidates = [
        Path(cfg.resolve("storage.results_dir")).parents[1] / "evaluations" / "v3.json",
        Path(cfg.resolve("storage.results_dir")).parents[1] / "evaluations" / "v1.json",
        Path(cfg.resolve("storage.predictions"))
    ]
    
    precomputed = []
    for candidate in eval_candidates:
        if candidate.exists():
            try:
                if candidate.suffix == ".json":
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    precomputed = data.get("results", [])
                    if precomputed:
                        break
            except Exception:
                continue

    if precomputed:
        options = {
            f"[{r.get('sector', 'other').upper()}] {r.get('sample_id', '').split('::')[-1]} "
            f"({', '.join(r.get('labels', []))} - {r.get('overall_confidence', 0):.0%})": r
            for r in precomputed[:50]
        }
        chosen = st.selectbox("Select a verified pre-run sample from the threat intelligence archive:", list(options.keys()))
        selected = options[chosen]

        image_path = selected.get("screenshot_path")
        sample_id = selected.get("sample_id")
        source = selected.get("source", "archive")

        # Reconstruct into AnalysisResult for rendering
        intentions = [
            IntentionResult(
                category=i.get("category", ""),
                confidence=i.get("confidence", 0.0),
                evidence=i.get("evidence", []),
                explanation=i.get("explanation", "")
            ) for i in selected.get("intentions", [])
        ]
        specialists = [
            SpecialistFinding(
                category=s.get("category", ""),
                confirmed=s.get("confirmed", False),
                confidence=s.get("confidence", 0.0),
                evidence=s.get("evidence", []),
                reasoning=s.get("reasoning", ""),
                model=s.get("model", "")
            ) for s in selected.get("specialist_findings", [])
        ]
        cat_scores = [
            CategoryScore(
                category=c.get("category", ""),
                score=c.get("score", 0.0),
                evidence=c.get("evidence", [])
            ) for c in selected.get("category_scores", [])
        ]
        steps = [
            AgentStep(
                agent=st_dict.get("agent", ""),
                layer=st_dict.get("layer", ""),
                model=st_dict.get("model", ""),
                stage=st_dict.get("stage", "framework"),
                vision=st_dict.get("vision", False),
                status=st_dict.get("status", "done"),
                started_at=st_dict.get("started_at"),
                ended_at=st_dict.get("ended_at"),
                summary=st_dict.get("summary", ""),
                output=st_dict.get("output", {}),
                error=st_dict.get("error")
            ) for st_dict in selected.get("steps", [])
        ]

        st.session_state["last_result"] = AnalysisResult(
            sample_id=selected.get("sample_id", "sample"),
            source=source,
            screenshot_path=image_path,
            intentions=intentions,
            sector=selected.get("sector", "other"),
            overall_confidence=selected.get("overall_confidence", 0.0),
            is_phishing=selected.get("is_phishing", True),
            phishing_score=selected.get("phishing_score", 0.0),
            verdict=selected.get("verdict", ""),
            risk_summary=selected.get("risk_summary", ""),
            recommended_action=selected.get("recommended_action", ""),
            steps=steps,
            specialist_findings=specialists,
            category_scores=cat_scores,
            vision_calls=selected.get("vision_calls", 0),
            elapsed=selected.get("elapsed", 0.0)
        )
    else:
        st.warning("No pre-computed evaluation records found in `evaluations/`.")

else:
    # PRO TIER - Uploading enabled
    st.success("💎 **Pro Plan Active**: Live screenshot analysis and multi-agent execution unlocked.")
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
# 2 · Pipeline Execution Panel
# ---------------------------------------------------------------------------
st.markdown("### 2 · Run the agent pipeline")
left, right = st.columns([1, 1.45])

with left:
    if image_path and Path(image_path).exists():
        st.image(image_path, caption=Path(image_path).name, **FULL_WIDTH)
    else:
        st.info("Awaiting screenshot selection.")

    mode = st.radio("Pipeline Mode", [
        "PhishIntentionLLM (5-layer multi-agent RAG)",
        "Single-agent baseline"
    ], index=0)

    save_result = st.checkbox("Save this prediction for evaluation", value=True)
    
    launch = st.button("🚀 Analyse intentions", 
                       type="primary",
                       disabled=not (is_pro and image_path and Path(image_path).exists()), 
                       **FULL_WIDTH)
    if not is_pro:
        st.caption("🔒 *Execution is disabled on Free tier. Sign in as Pro to run live models.*")

with right:
    flow_placeholder = st.empty()
    initial_steps = st.session_state.get("last_result").steps if st.session_state.get("last_result") else []
    flow_placeholder.markdown(render_agent_flow(initial_steps), unsafe_allow_html=True)
    flow_legend("framework")

# Live execution for authenticated Pro users
if launch and is_pro and image_path:
    live_steps = []

    def on_step(step):
        matched = False
        for idx, s in enumerate(live_steps):
            if s.agent == step.agent and s.layer == step.layer:
                live_steps[idx] = step
                matched = True
                break
        if not matched:
            live_steps.append(step)

        flow_placeholder.markdown(
            render_agent_flow(live_steps, compact=False, title="Live Multi-Agent Pipeline Execution"),
            unsafe_allow_html=True
        )

    status_widget = st.status("🤖 Multi-agent framework running...", expanded=True)
    with status_widget:
        if mode.startswith("PhishIntentionLLM"):
            result = get_framework().analyse_screenshot(
                image_path, on_step=on_step, brand_hint=brand_hint,
                sample_id=sample_id, source=source)
        else:
            from phishintentionllm.pipeline import SingleAgentBaseline
            engine = SingleAgentBaseline(cfg, get_framework().client)
            result = engine.analyse_screenshot(image_path, on_step=on_step,
                                               sample_id=sample_id, source=source)
        status_widget.update(label="✅ Analysis complete!", state="complete", expanded=False)

    flow_placeholder.markdown(render_agent_flow(result.steps, compact=False), unsafe_allow_html=True)
    st.session_state["last_result"] = result
    if save_result and not result.error:
        store.save_prediction(result)

# ---------------------------------------------------------------------------
# 3 · Assessment Display
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