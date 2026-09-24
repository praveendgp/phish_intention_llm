"""✍️ Manual Annotation - the human labelling protocol of the paper.

Two roles in one page:
  1. Label samples from scratch (two engineers + a reviewer).
  2. Verify or correct rows the annotator ensemble wrote into the manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, category_style, footer,
                           get_config, get_manual_manager, get_registry,
                           get_store, hero, render_label_chips, render_votes,
                           sidebar_status, style_page)
from phishintentionllm.schemas import Intention

style_page()
cfg = get_config()
registry = get_registry()
store = get_store()
manager = get_manual_manager()

hero("Manual Annotation",
     "Human labelling, preserved exactly as the project intends: two independent "
     "engineers plus a reviewer — and the place where annotator-generated "
     "manifest rows get human sign-off.",
     tags=["2 labellers + 1 reviewer", "Multi-label", "Verifies the manifest",
           "Builds the ground truth"])
sidebar_status()

if not manager.enabled:
    st.warning("Manual annotation is disabled in `config.yaml`.")
    footer()
    st.stop()

existing = [d["source"] for d in registry.available() if d["exists"]]
if not existing:
    st.warning("No dataset available. Add data under `data/raw/` first.")
    footer()
    st.stop()

st.markdown("### 1 · Pick your identity and a queue")
col_a, col_b, col_c = st.columns(3)
with col_a:
    annotator = st.selectbox("I am", manager.annotators)
with col_b:
    source = st.selectbox("Source", existing)
with col_c:
    limit = st.number_input("Load first N", 20, 20000, 200, step=20)

samples = registry.load_source(source, limit=int(limit))
if not samples:
    st.warning("No samples found.")
    footer()
    st.stop()

manifest_map = store.manifest_map()
manifest_rows = {r["sample_id"]: r for r in store.load_manifest_rows()}

filter_mode = st.radio(
    "Queue",
    ["All", "Unlabelled by me", "Awaiting review", "Reviewed",
     "⚠️ Manifest flagged for review"],
    horizontal=True)


def keep(sample) -> bool:
    if filter_mode == "⚠️ Manifest flagged for review":
        return bool(manifest_rows.get(sample.sample_id, {}).get("needs_review"))
    status = manager.status(sample.sample_id)
    if filter_mode == "All":
        return True
    if filter_mode == "Unlabelled by me":
        return annotator not in status["records"]
    if filter_mode == "Awaiting review":
        return status["state"] == "awaiting review"
    return status["state"] == "reviewed"


queue = [s for s in samples if keep(s)]
if not queue:
    st.success("Nothing left in this queue. 🎉")
    footer()
    st.stop()

if "manual_index" not in st.session_state:
    st.session_state["manual_index"] = 0
st.session_state["manual_index"] %= len(queue)
index = st.session_state["manual_index"]
sample = queue[index]
manifest_row = manifest_rows.get(sample.sample_id)

st.caption(f"Sample {index + 1} of {len(queue)} in this queue.")

# ---------------------------------------------------------------------------
st.markdown("### 2 · Label the screenshot")
image_col, form_col = st.columns([1.15, 1])

with image_col:
    try:
        st.image(sample.screenshot_path, **FULL_WIDTH)
    except Exception as exc:
        st.error(f"Cannot display image: {exc}")
    st.code(sample.sample_id, language="text")
    if sample.brand:
        st.caption(f"Dataset brand hint: **{sample.brand}**")

    navigation = st.columns(2)
    if navigation[0].button("⬅️ Previous", **FULL_WIDTH):
        st.session_state["manual_index"] = (index - 1) % len(queue)
        st.rerun()
    if navigation[1].button("Skip ➡️", **FULL_WIDTH):
        st.session_state["manual_index"] = (index + 1) % len(queue)
        st.rerun()

    if manifest_row:
        st.markdown("#### 📋 What the annotators decided")
        render_label_chips(manifest_row.get("labels", []), "Manifest:")
        st.caption(f"Agreement: **{manifest_row.get('agreement')}** · tie-break: "
                   f"**{manifest_row.get('tie_break_used')}** · verified: "
                   f"**{manifest_row.get('human_verified')}**")
        if manifest_row.get("needs_review"):
            st.warning("Flagged for review: "
                       + ", ".join(manifest_row.get("review_reasons", [])))
        with st.expander("Annotator votes"):
            render_votes(manifest_row.get("votes", []))

with form_col:
    status = manager.status(sample.sample_id)
    state_icon = {"reviewed": "🟢", "awaiting review": "🟡",
                  "partially labelled": "🟠", "unlabelled": "⚪"}
    st.markdown(f"**Status:** {state_icon.get(status['state'], '⚪')} {status['state']}")
    if status["annotator_agreement"] is not None:
        st.caption("Labellers agree ✅" if status["annotator_agreement"]
                   else "Labellers disagree ❗ — reviewer decision required")

    existing_record = status["records"].get(annotator, {})
    default = existing_record.get("categories") or []

    st.markdown("**Intentions** (select one or more)")
    picked = []
    for intention in Intention.all():
        style = category_style(intention.value)
        if st.checkbox(f"{style['icon']} {intention.value} — {style['plain']}",
                       value=intention.value in default,
                       key=f"cb_{sample.sample_id}_{intention.value}_{annotator}"):
            picked.append(intention.value)

    sectors = cfg.sectors
    current_sector = existing_record.get("sector") or (
        manifest_row.get("sector") if manifest_row else "other")
    sector = st.selectbox("Sector", sectors,
                          index=sectors.index(current_sector)
                          if current_sector in sectors else sectors.index("other"))
    is_phishing = st.checkbox("This page is phishing", value=sample.is_phishing)
    notes = st.text_area("Notes / justification",
                         value=existing_record.get("notes", ""), height=90)

    save_col, next_col = st.columns(2)
    if save_col.button("💾 Save label", type="primary", **FULL_WIDTH):
        if not picked and is_phishing:
            st.error("Select at least one intention for a phishing page.")
        else:
            manager.annotate(sample.sample_id, annotator, picked, sector,
                             notes, is_phishing)
            st.success(f"Saved as **{annotator}**.")
            st.rerun()
    if next_col.button("💾 Save & next ➡️", **FULL_WIDTH):
        if picked or not is_phishing:
            manager.annotate(sample.sample_id, annotator, picked, sector,
                             notes, is_phishing)
            st.session_state["manual_index"] = (index + 1) % len(queue)
            st.rerun()
        else:
            st.error("Select at least one intention for a phishing page.")

    if manifest_row and annotator == manager.reviewer:
        st.divider()
        st.markdown("**Manifest verification** (reviewer only)")
        st.caption("Write your selection above back into the manifest and mark "
                   "the row as human-verified.")
        if st.button("✅ Verify / correct manifest row", **FULL_WIDTH):
            if manager.verify_manifest_row(sample.sample_id, picked,
                                           reviewer=annotator, sector=sector,
                                           notes=notes):
                st.success("Manifest row updated and marked human-verified.")
                st.rerun()
            else:
                st.error("This sample is not in the manifest yet.")

# ---------------------------------------------------------------------------
st.markdown("### 3 · Submissions for this sample")
records = manager.status(sample.sample_id)["records"]
if records:
    st.dataframe([{"Annotator": who,
                   "Intentions": ", ".join(r.get("categories", [])) or "—",
                   "Sector": r.get("sector", "—"),
                   "Reviewed": r.get("reviewed", False),
                   "Notes": (r.get("notes") or "")[:90]}
                  for who, r in records.items()], hide_index=True, **FULL_WIDTH)
    labels, basis = manager.consensus(sample.sample_id)
    st.info(f"**Human consensus ({basis}):** {', '.join(labels) or 'none yet'}")
else:
    st.caption("No human submissions yet for this sample.")

# ---------------------------------------------------------------------------
st.markdown("### 4 · Ground-truth dataset")
distribution = manager.distribution("ground_truth")
metrics = st.columns(4)
metrics[0].metric("Reviewed records", distribution["total"])
metrics[1].metric("Manual submissions", len(store.load_manual()))
metrics[2].metric("Exported ground truth", len(store.load_ground_truth()))
metrics[3].metric("Manifest verified",
                  sum(1 for r in manifest_rows.values() if r.get("human_verified")))

if distribution["per_category"]:
    dist_cols = st.columns(2)
    with dist_cols[0]:
        st.caption("Intentions per category")
        bar_chart(distribution["per_category"])
    with dist_cols[1]:
        st.caption("Intentions per record")
        bar_chart(distribution["per_intention_count"], horizontal=False)

export_col, require_col = st.columns(2)
require_review = require_col.checkbox("Only export reviewed samples",
                                      value=manager.require_reviewer)
if export_col.button("📦 Export ground truth", type="primary", **FULL_WIDTH):
    rows = manager.build_ground_truth(samples, require_review=require_review)
    st.success(f"Exported {store.write_ground_truth(rows)} record(s) to "
               f"`{store.gt_path}`.")

footer()
