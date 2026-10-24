"""
Intent Scanner View - Single-Agent and Multi-Agent RAG Phishing Detection.
Subscription Gated:
- Free Tier: Pre-computed Threat Intelligence cases only.
- Premium Tier: Live Single-Agent and 5-Layer Multi-Agent RAG execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

from phishintentionllm.auth_clerk import is_premium_user, get_current_user_tier
from phishintentionllm.pipeline import SingleAgentBaseline
from phishintentionllm.schemas import AnalysisResult, IntentionResult
from ui_components import (
    FULL_WIDTH, flow_legend, get_config, get_framework, get_store, hero,
    render_agent_flow, render_intentions, render_label_chips,
    render_specialists, render_verdict, sidebar_status, style_page
)

style_page()
cfg = get_config()
store = get_store()

hero("Intent Scanner", "Analyse login-page screenshots for deceptive intent",
     tags=["5-layer multi-agent", "CoT intention classes", "Clerk Subscription RBAC", "MTech Project"])

sidebar_status()

is_pro = is_premium_user()
tier = get_current_user_tier()


def get_threat_samples(store_obj: Any, max_limit: int = 25) -> List[Dict[str, Any]]:
    """Defensively retrieve benchmark manifest samples without attribute crashes."""
    samples: List[Dict[str, Any]] = []

    # 1. Try store.list_samples if method exists
    if hasattr(store_obj, "list_samples"):
        try:
            raw = store_obj.list_samples(limit=max_limit)
            if raw:
                for item in raw:
                    if isinstance(item, dict):
                        samples.append({
                            "id": item.get("id") or item.get("sample_id") or "sample",
                            "brand": item.get("brand") or "Unknown",
                            "source": item.get("source") or "benchmark",
                            "image_path": item.get("image_path") or item.get("screenshot_path") or "",
                            "ground_truth_intention": item.get("ground_truth_intention") or ", ".join(item.get("labels", [])) or "unlabeled",
                        })
                    else:
                        samples.append({
                            "id": getattr(item, "sample_id", getattr(item, "id", "sample")),
                            "brand": getattr(item, "brand", "Unknown") or "Unknown",
                            "source": getattr(item, "source", "benchmark") or "benchmark",
                            "image_path": getattr(item, "screenshot_path", getattr(item, "image_path", "")),
                            "ground_truth_intention": getattr(item, "ground_truth_intention", None) or ", ".join(getattr(item, "labels", [])) or "unlabeled",
                        })
        except Exception:
            pass

    # 2. Try store.load_manifest_rows()
    if not samples and hasattr(store_obj, "load_manifest_rows"):
        try:
            rows = store_obj.load_manifest_rows()
            for r in rows[:max_limit]:
                samples.append({
                    "id": r.get("sample_id", "sample"),
                    "brand": r.get("brand") or "Unknown",
                    "source": r.get("source") or "manifest",
                    "image_path": r.get("screenshot_path", ""),
                    "ground_truth_intention": ", ".join(r.get("labels", [])) if r.get("labels") else "unlabeled",
                })
        except Exception:
            pass

    # 3. Try store.load_manifest()
    if not samples and hasattr(store_obj, "load_manifest"):
        try:
            m_list = store_obj.load_manifest()
            for m in m_list[:max_limit]:
                samples.append({
                    "id": getattr(m, "sample_id", "sample"),
                    "brand": getattr(m, "brand", "Unknown") or "Unknown",
                    "source": getattr(m, "source", "manifest") or "manifest",
                    "image_path": getattr(m, "screenshot_path", ""),
                    "ground_truth_intention": ", ".join(getattr(m, "labels", [])) if getattr(m, "labels", None) else "unlabeled",
                })
        except Exception:
            pass

    return samples


# ---------------------------------------------------------------------------
# 1 · Input Selection (Gated by Subscription Tier)
# ---------------------------------------------------------------------------
st.markdown("### 1 · Input Selection")

image_path: Optional[str] = None
brand_hint: Optional[str] = None
sample_id: Optional[str] = None
source = "upload"

if not is_pro:
    st.info("ℹ️ **Community Free Tier**: You can inspect pre-computed evaluation cases from the threat archive. **Upgrade to Premium** in the sidebar to upload custom screenshots and run real-time inference.")
    tab_sample = st.tabs(["📁 Threat Intelligence Samples (Free & Pro)"])[0]
    with tab_sample:
        samples = get_threat_samples(store, max_limit=25)
        if not samples:
            st.info("No pre-computed manifest records found in `data/outputs/manifest.jsonl` yet.")
            st.markdown("To populate benchmark samples, run: `python -m phishintentionllm.cli annotate --source dataset` or upgrade to **Premium** in the sidebar to upload screenshots directly.")
        else:
            sample_labels = [f"{s['id']} — {s['brand']} ({s['source']})" for s in samples]
            chosen_idx = st.selectbox("Select pre-computed sample", range(len(samples)), format_func=lambda i: sample_labels[i])
            chosen = samples[chosen_idx]
            image_path = chosen["image_path"]
            brand_hint = chosen["brand"]
            sample_id = chosen["id"]
            source = chosen["source"]
            st.caption(f"Brand: **{chosen['brand']}** | Ground truth intention: `{chosen['ground_truth_intention']}`")
else:
    st.success("💎 **Premium Tier Active**: Real-time multi-agent RAG pipeline unlocked.")
    tab_upload, tab_sample = st.tabs(["📤 Upload screenshot (Premium)", "📁 Choose from sample library"])
    with tab_upload:
        up = st.file_uploader("Screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
        brand_hint_in = st.text_input("Target brand (optional hint)", placeholder="e.g. PayPal, Bank of America, Microsoft")
        if up is not None:
            tmp_dir = Path("data/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            saved = tmp_dir / up.name
            saved.write_bytes(up.getbuffer())
            image_path = str(saved)
            brand_hint = brand_hint_in.strip() or None
            sample_id = saved.stem
            source = "upload"

    with tab_sample:
        samples = get_threat_samples(store, max_limit=25)
        if samples:
            sample_labels = [f"{s['id']} — {s['brand']} ({s['source']})" for s in samples]
            chosen_idx = st.selectbox("Select sample", range(len(samples)), format_func=lambda i: sample_labels[i], key="pro_sample_sel")
            chosen = samples[chosen_idx]
            if not image_path:
                image_path = chosen["image_path"]
                brand_hint = chosen["brand"]
                sample_id = chosen["id"]
                source = chosen["source"]
        else:
            st.caption("No pre-computed samples in manifest yet. Upload a screenshot above.")

# ---------------------------------------------------------------------------
# 2 · Execution Configuration & RBAC Gate
# ---------------------------------------------------------------------------
st.markdown("### 2 · Execution Configuration")
left, right = st.columns([1, 1])

with left:
    if image_path and Path(image_path).exists():
        st.image(image_path, caption=f"Target: {Path(image_path).name}", use_container_width=True)
    elif image_path:
        st.caption(f"Target selected: `{image_path}`")
    else:
        st.info("Select or upload an image to preview.")

    mode = st.radio(
        "Detection Engine",
        ["PhishIntentionLLM (5-layer Multi-Agent RAG)", "Single-agent baseline"],
        index=0,
        disabled=not is_pro
    )
    save_result = st.checkbox("Save this prediction for evaluation", value=True)

    launch = st.button(
        "🚀 Analyse intentions",
        type="primary",
        disabled=not (is_pro and image_path and Path(image_path).exists()),
        **FULL_WIDTH
    )

    if not is_pro:
        st.warning("🔒 **Model Execution Locked**: Single-Agent and Multi-Agent RAG inference require a **Premium Subscription**.")
        st.caption("Click **Upgrade to Premium (Clerk API)** in the left sidebar to unlock live execution.")

with right:
    flow_placeholder = st.empty()
    initial_steps = st.session_state.get("last_result").steps if st.session_state.get("last_result") else []
    flow_placeholder.markdown(render_agent_flow(initial_steps), unsafe_allow_html=True)
    flow_legend("framework")

# ---------------------------------------------------------------------------
# 3 · Live Execution (Premium Only)
# ---------------------------------------------------------------------------
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
                sample_id=sample_id, source=source
            )
        else:
            baseline = SingleAgentBaseline(cfg, get_framework().client)
            result = baseline.analyse_screenshot(image_path, brand_hint=brand_hint)

        if save_result:
            store.save_prediction(result)

        st.session_state["last_result"] = result
        status_widget.update(label="Analysis complete!", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# 4 · Display Results (Clean result.intentions & result.labels)
# ---------------------------------------------------------------------------
result: Optional[AnalysisResult] = st.session_state.get("last_result")
if result:
    st.markdown("### 3 · Analysis Verdict & Intention Breakdown")
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        render_verdict(result)
        render_intentions(result.intentions)
        render_specialists(result)
    with col_right:
        st.markdown("#### Agent Execution History")
        st.markdown(render_agent_flow(result.steps, compact=False, title="Pipeline Trace"), unsafe_allow_html=True)
        render_label_chips(result.labels, title="Predicted Intentions:")
