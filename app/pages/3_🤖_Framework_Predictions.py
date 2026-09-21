"""Run the five-layer framework over many samples and profile the predictions."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, category_style, flow_legend,
                           footer, get_config, get_framework, get_registry,
                           get_store, hero, render_agent_flow, result_summary_row,
                           setup_page, sidebar_status)

setup_page("Framework Predictions", "🤖")
cfg = get_config()
registry = get_registry()
store = get_store()

hero("Framework Predictions",
     "The system under test: five vision-language layers analyse each screenshot "
     "and produce predictions, which are then scored against the manifest.",
     tags=["5 layers, all VLM", "Never sees the manifest", "Checkpointed",
           "Sector × intention profiling"])
sidebar_status()

existing = [d["source"] for d in registry.available() if d["exists"]]
if not existing:
    st.warning("No dataset available. Add data under `data/raw/` first.")
    footer()
    st.stop()

st.markdown("### Configure the run")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    sources = st.multiselect("Sources", existing, default=existing)
with col_b:
    count = st.number_input("Samples", 1, 2000, 10)
with col_c:
    mode = st.selectbox("Pipeline", ["framework", "single-agent"])
with col_d:
    scope = st.selectbox("Sample pool",
                         ["manifest-covered only", "random (stratified)", "first N"])

skip_done = st.checkbox("Skip samples already predicted", value=True)

if not sources:
    st.info("Pick at least one source.")
    footer()
    st.stop()

manifest_ids = set(store.manifest_ids())
if scope == "manifest-covered only":
    if not manifest_ids:
        st.warning("The manifest is empty — build it first on the "
                   "**Manifest Builder** page, or pick another pool.")
        pool = []
    else:
        pool = [s for s in registry.load_all(sources=sources)
                if s.sample_id in manifest_ids]
        st.caption("Restricting to samples that already have reference labels, "
                   "so the run is immediately scoreable.")
elif scope.startswith("random"):
    pool = registry.sample(int(count) * 3, sources=sources, stratify=True)
else:
    pool = registry.load_all(limit_per_source=int(count), sources=sources)

if skip_done:
    predicted = set(store.prediction_map())
    pool = [s for s in pool if s.sample_id not in predicted]

selection = pool[: int(count)]
covered = sum(1 for s in selection if s.sample_id in manifest_ids)
st.caption(f"{len(selection)} sample(s) queued · {covered} of them have manifest "
           f"reference labels.")

if "batch_running" not in st.session_state:
    st.session_state["batch_running"] = False

start_col, stop_col = st.columns(2)
start = start_col.button("▶️ Start predictions", type="primary",
                         disabled=not selection, **FULL_WIDTH)
if stop_col.button("⏹️ Stop", **FULL_WIDTH):
    st.session_state["batch_running"] = False

flow_legend("framework")

if start:
    st.session_state["batch_running"] = True
    from phishintentionllm.pipeline import BatchRunner

    runner = BatchRunner(cfg, get_framework(), store)
    progress = st.progress(0.0, text="Starting…")
    live_flow = st.empty()
    live_table = st.empty()
    rows: list = []
    started = time.time()
    buffer: list = []

    def on_step(step):
        buffer.append(step)
        live_flow.markdown(render_agent_flow(buffer[-4:], compact=True),
                           unsafe_allow_html=True)

    def on_progress(index, total, result):
        buffer.clear()
        rows.append(result_summary_row(result))
        eta = (time.time() - started) / index * (total - index)
        progress.progress(index / max(1, total),
                          text=f"{index}/{total} · ETA {eta/60:.1f} min")
        live_table.dataframe(rows[::-1], hide_index=True, **FULL_WIDTH)

    payload = runner.run(
        selection, mode=("framework" if mode == "framework" else "single"),
        on_progress=on_progress, on_step=on_step,
        stop_flag=lambda: not st.session_state.get("batch_running", True))

    st.session_state["batch_running"] = False
    st.session_state["last_batch"] = payload
    progress.progress(1.0, text="Completed")
    st.success(f"Run `{payload['run_id']}` finished — {payload['n_success']} "
               f"succeeded, {payload['n_failed']} failed.")

payload = st.session_state.get("last_batch")
if payload:
    st.markdown("### Run summary")
    metrics = st.columns(5)
    metrics[0].metric("Predicted", payload["n_success"])
    metrics[1].metric("Failed", payload["n_failed"])
    metrics[2].metric("Avg / sample", f"{payload['avg_seconds_per_sample']}s")
    metrics[3].metric("VLM calls", payload["total_vision_calls"])
    metrics[4].metric("Scored vs manifest", payload["n_scored_against_manifest"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Predicted intention distribution")
        bar_chart(payload["per_category"])
        for category, value in sorted(payload["per_category"].items(),
                                      key=lambda kv: -kv[1]):
            style = category_style(category)
            st.caption(f"{style['icon']} **{category}** — {value} sample(s)")
    with right:
        st.markdown("#### Intention combinations")
        if payload["co_occurrence"]:
            bar_chart(payload["co_occurrence"])
        else:
            st.caption("No multi-intention samples in this run.")

    st.markdown("#### Sector × intention matrix")
    matrix = payload.get("sector_matrix") or {}
    if matrix:
        st.dataframe([{"Sector": s, **c} for s, c in matrix.items()],
                     hide_index=True, **FULL_WIDTH)

    if payload.get("evaluation"):
        st.markdown("#### Immediate score against the manifest")
        report = payload["evaluation"]
        score_cols = st.columns(5)
        score_cols[0].metric("Precision", f"{report['precision_micro']:.4f}")
        score_cols[1].metric("Recall", f"{report['recall_micro']:.4f}")
        score_cols[2].metric("F1", f"{report['f1_micro']:.4f}")
        score_cols[3].metric("Accuracy (micro)", f"{report['accuracy_micro']:.4f}")
        score_cols[4].metric("Overall accuracy", f"{report['overall_accuracy']:.4f}")
        st.caption("Full breakdown on the **Evaluation** page.")

    with st.expander("All results in this run"):
        st.dataframe([{"sample_id": r["sample_id"],
                       "labels": ", ".join(r["labels"]), "sector": r["sector"],
                       "confidence": r["overall_confidence"],
                       "verdict": r["verdict"], "error": r.get("error") or ""}
                      for r in payload["results"]], hide_index=True, **FULL_WIDTH)

st.markdown("### Previous prediction runs")
runs = store.list_runs()
if not runs:
    st.caption("No saved runs yet.")
else:
    chosen = st.selectbox("Saved run", runs, format_func=lambda p: p.name)
    if st.button("Load run"):
        st.session_state["last_batch"] = store.load_run(chosen)
        st.rerun()

footer()
