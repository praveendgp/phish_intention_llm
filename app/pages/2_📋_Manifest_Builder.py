"""Manifest Builder - the annotator ensemble that creates the reference labels.

This is the stage the project brief assigns to the annotators: two independent
vision-language models label each screenshot, a tie-breaker resolves genuine
disputes, and a finaliser writes one manifest row per sample. The resulting
`manifest.jsonl` is what the framework's predictions are evaluated against.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, category_style, flow_legend,
                           footer, get_config, get_manifest_builder, get_registry,
                           get_store, hero, manifest_summary_row,
                           render_agent_flow, render_label_chips, render_votes,
                           setup_page, sidebar_status)

setup_page("Manifest Builder", "📋")
cfg = get_config()
registry = get_registry()
store = get_store()

hero("Manifest Builder",
     "Two independent VLM annotators label every screenshot, a tie-breaker "
     "settles disputes and a finaliser signs off — producing the "
     "<b>manifest</b> that the framework is later scored against.",
     tags=["Annotator A + Annotator B", "Tie-breaker on dispute",
           "Finaliser sign-off", "→ manifest.jsonl"])
sidebar_status()

existing = [d["source"] for d in registry.available() if d["exists"]]
if not existing:
    st.warning("No dataset available. Add data under `data/raw/` first.")
    footer()
    st.stop()

# ---------------------------------------------------------------------------
tab_build, tab_inspect = st.tabs(["🏗️ Build manifest", "🔎 Inspect manifest"])

# ===========================================================================
with tab_build:
    st.markdown("### Configure the annotation run")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        sources = st.multiselect("Sources", existing, default=existing)
    with col_b:
        count = st.number_input("Samples", 1, 2000, 10)
    with col_c:
        strategy = st.selectbox("Selection", ["random (stratified)", "first N"])
    with col_d:
        skip_done = st.checkbox("Skip already-annotated", value=True)

    if not sources:
        st.info("Pick at least one source.")
    else:
        if strategy.startswith("random"):
            pool = registry.sample(int(count) * 3, sources=sources, stratify=True)
        else:
            pool = registry.load_all(limit_per_source=int(count), sources=sources)

        if skip_done:
            already = set(store.manifest_ids())
            pool = [s for s in pool if s.sample_id not in already]

        selection = pool[: int(count)]
        st.caption(f"{len(selection)} sample(s) queued for annotation. "
                   f"Manifest currently holds {len(store.manifest_ids())} record(s).")

        if "manifest_running" not in st.session_state:
            st.session_state["manifest_running"] = False

        start_col, stop_col = st.columns(2)
        start = start_col.button("▶️ Build manifest", type="primary",
                                 disabled=not selection, **FULL_WIDTH)
        if stop_col.button("⏹️ Stop", **FULL_WIDTH):
            st.session_state["manifest_running"] = False

        flow_legend("manifest")

        if start:
            st.session_state["manifest_running"] = True
            builder = get_manifest_builder()
            progress = st.progress(0.0, text="Starting…")
            live_flow = st.empty()
            live_table = st.empty()
            rows: list = []
            started = time.time()
            steps_buffer: list = []

            def on_step(step):
                steps_buffer.append(step)
                live_flow.markdown(
                    render_agent_flow(steps_buffer[-4:], compact=True,
                                      title="Annotator Flow"),
                    unsafe_allow_html=True)

            def on_progress(index, total, record):
                steps_buffer.clear()
                rows.append(manifest_summary_row(record))
                eta = (time.time() - started) / index * (total - index)
                progress.progress(index / max(1, total),
                                  text=f"{index}/{total} · ETA {eta/60:.1f} min")
                live_table.dataframe(rows[::-1], hide_index=True, **FULL_WIDTH)

            payload = builder.build(
                selection, on_progress=on_progress, on_step=on_step,
                stop_flag=lambda: not st.session_state.get("manifest_running", True),
                overwrite=not skip_done)

            st.session_state["manifest_running"] = False
            st.session_state["last_manifest_run"] = payload
            progress.progress(1.0, text="Completed")
            st.success(f"Manifest run `{payload['manifest_run_id']}` finished — "
                       f"{payload['n_success']} annotated, {payload['n_failed']} failed.")

    # -----------------------------------------------------------------------
    payload = st.session_state.get("last_manifest_run")
    if payload:
        st.markdown("### Run summary")
        metrics = st.columns(5)
        metrics[0].metric("Annotated", payload["n_success"])
        metrics[1].metric("Failed", payload["n_failed"])
        metrics[2].metric("Avg / sample", f"{payload['avg_seconds_per_sample']}s")
        metrics[3].metric("Tie-breaks", payload["tie_breaks"])
        metrics[4].metric("Flagged for review", payload["needs_review"])

        left, right = st.columns(2)
        with left:
            st.markdown("#### Reference label distribution")
            bar_chart(payload["per_category"])
            for category, value in sorted(payload["per_category"].items(),
                                          key=lambda kv: -kv[1]):
                style = category_style(category)
                st.caption(f"{style['icon']} **{category}** — {value} sample(s)")
        with right:
            st.markdown("#### Annotator agreement")
            bar_chart(payload["agreement_breakdown"])
            st.markdown("#### Intentions per record")
            bar_chart(payload["per_intention_count"], horizontal=False)

        st.caption("Annotator models used: "
                   + " · ".join(f"`{k}`: {v}" for k, v in
                                payload["annotator_models"].items()))

        with st.expander("Records produced in this run"):
            st.dataframe(
                [{"sample_id": r["sample_id"], "labels": ", ".join(r["labels"]),
                  "agreement": r["agreement"], "tie_break": r["tie_break_used"],
                  "needs_review": r["needs_review"], "sector": r["sector"],
                  "error": r.get("error") or ""} for r in payload["records"]],
                hide_index=True, **FULL_WIDTH)

    st.markdown("### Previous manifest runs")
    runs = store.list_manifest_runs()
    if not runs:
        st.caption("No saved manifest runs yet.")
    else:
        chosen = st.selectbox("Saved run", runs, format_func=lambda p: p.name)
        if st.button("Load manifest run"):
            st.session_state["last_manifest_run"] = store.load_run(chosen)
            st.rerun()

# ===========================================================================
with tab_inspect:
    rows = store.load_manifest_rows()
    if not rows:
        st.info("The manifest is empty. Build it on the previous tab.")
    else:
        st.markdown("### Manifest overview")
        metrics = st.columns(5)
        metrics[0].metric("Records", len(rows))
        metrics[1].metric("Multi-intention",
                          sum(1 for r in rows if len(r.get("labels", [])) > 1))
        metrics[2].metric("Tie-breaks",
                          sum(1 for r in rows if r.get("tie_break_used")))
        metrics[3].metric("Needs review",
                          sum(1 for r in rows if r.get("needs_review")))
        metrics[4].metric("Human verified",
                          sum(1 for r in rows if r.get("human_verified")))

        filter_cols = st.columns([1, 1, 2])
        with filter_cols[0]:
            only_review = st.checkbox("Only flagged for review")
        with filter_cols[1]:
            label_filter = st.selectbox("Filter by label", ["(all)"] + cfg.categories)
        with filter_cols[2]:
            search = st.text_input("Search sample id")

        filtered = [
            r for r in rows
            if (not only_review or r.get("needs_review"))
            and (label_filter == "(all)" or label_filter in r.get("labels", []))
            and (not search or search.lower() in r.get("sample_id", "").lower())
        ]
        st.caption(f"{len(filtered)} record(s) after filtering.")

        st.dataframe([manifest_summary_row(r) for r in filtered[:300]],
                     hide_index=True, **FULL_WIDTH)

        if filtered:
            st.markdown("### Inspect a record")
            choice = st.selectbox("Record", range(len(filtered)),
                                  format_func=lambda i: filtered[i]["sample_id"])
            record = filtered[choice]

            img_col, detail_col = st.columns([1, 1.2])
            with img_col:
                try:
                    st.image(record["screenshot_path"], **FULL_WIDTH)
                except Exception:
                    st.caption("⚠️ screenshot unavailable")
            with detail_col:
                render_label_chips(record.get("labels", []), "Manifest labels:")
                st.caption(f"Agreement: **{record.get('agreement')}** · "
                           f"tie-break: **{record.get('tie_break_used')}** · "
                           f"sector: **{record.get('sector')}**")
                if record.get("needs_review"):
                    st.warning("Flagged for human review: "
                               + ", ".join(record.get("review_reasons", [])))
                if record.get("human_verified"):
                    st.success("Human verified.")
                if record.get("finalizer_summary"):
                    st.markdown(f"**Finaliser:** {record['finalizer_summary']}")
                for category, evidence in (record.get("evidence") or {}).items():
                    if evidence:
                        st.markdown(f"**{category} evidence**")
                        for item in evidence[:4]:
                            st.markdown(f"- {item}")

            st.markdown("#### Independent annotator votes")
            render_votes(record.get("votes", []))

            with st.expander("Annotator agent trace"):
                st.markdown(render_agent_flow(record.get("steps", []),
                                              title="Annotator Flow"),
                            unsafe_allow_html=True)
            with st.expander("Raw manifest row"):
                st.json(record)

        st.markdown("### Export")
        export_cols = st.columns(2)
        if export_cols[0].button("⬇️ Export manifest as CSV", **FULL_WIDTH):
            path = store.export_manifest_csv(
                cfg.resolve("storage.results_dir") / "manifest.csv")
            st.success(f"Written to `{path}`")
        export_cols[1].download_button(
            "⬇️ Download manifest.jsonl",
            data="\n".join(__import__("json").dumps(r) for r in rows),
            file_name="manifest.jsonl", mime="application/jsonl", **FULL_WIDTH)

footer()
