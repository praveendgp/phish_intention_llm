"""Evaluation dashboard.

Framework predictions are scored against the annotator-generated **manifest**
(or, optionally, the human-reviewed ground truth), using the metric suite of
Section 5.3 of the base paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, footer, get_config,
                           get_evaluator, get_store, hero, setup_page,
                           sidebar_status)

setup_page("Evaluation", "📊")
cfg = get_config()
store = get_store()
evaluator = get_evaluator()

hero("Evaluation",
     "How well does the framework reproduce the annotators' manifest? "
     "Micro-averaged precision/recall/F1, accuracy by complexity, per-class "
     "metrics and error analysis.",
     tags=["Reference = manifest", "Eq. 3-10 of the paper",
           "Acc_comp partial matching", "Framework vs single-agent"])
sidebar_status()

manifest_rows = store.load_manifest_rows()
predictions = store.load_predictions()

metrics = st.columns(4)
metrics[0].metric("Manifest records", len(manifest_rows))
metrics[1].metric("Stored predictions", len(predictions))
metrics[2].metric("Ground-truth records", len(store.load_ground_truth()))
metrics[3].metric("Saved runs", len(store.list_runs()))

if not manifest_rows and not store.load_ground_truth():
    st.warning("No reference labels yet. Build the manifest on the "
               "**Manifest Builder** page first.")
    footer()
    st.stop()

# ---------------------------------------------------------------------------
st.markdown("### Scope")
scope_cols = st.columns([1.4, 1, 1])
runs = store.list_runs()
options = ["All stored predictions"] + [p.name for p in runs]
with scope_cols[0]:
    choice = st.selectbox("Evaluate", options)
with scope_cols[1]:
    reference = st.selectbox(
        "Reference labels", ["manifest", "ground_truth"],
        index=0 if cfg.get("evaluation.reference") == "manifest" else 1,
        help="manifest = annotator ensemble output; "
             "ground_truth = human-reviewed labels")
with scope_cols[2]:
    verified_only = st.checkbox("Human-verified rows only", value=False,
                                help="Restrict to manifest rows a reviewer signed off")

run_path = None if choice == options[0] else str(
    next(p for p in runs if p.name == choice))

outcome = evaluator.evaluate_run(run_path, reference, verified_only)
if "error" in outcome:
    st.error(outcome["error"])
    st.caption(f"{outcome['n_reference']} reference record(s), "
               f"{outcome['n_predictions']} prediction(s) — no overlapping sample ids.")
    footer()
    st.stop()

report = outcome["report"]

# ---------------------------------------------------------------------------
st.markdown("### General performance")
st.caption(f"Scored **{outcome['n_evaluated']}** sample(s) that have both a "
           f"`{outcome['reference']}` label and a framework prediction.")

cols = st.columns(5)
cols[0].metric("Precision (micro)", f"{report['precision_micro']:.4f}")
cols[1].metric("Recall (micro)", f"{report['recall_micro']:.4f}")
cols[2].metric("F1 (micro)", f"{report['f1_micro']:.4f}")
cols[3].metric("Accuracy (micro)", f"{report['accuracy_micro']:.4f}")
cols[4].metric("Overall accuracy", f"{report['overall_accuracy']:.4f}")

chart_col, comp_col = st.columns(2)
with chart_col:
    st.markdown("#### Micro-averaged metrics")
    bar_chart({"Precision": report["precision_micro"],
               "Recall": report["recall_micro"],
               "F1": report["f1_micro"],
               "Accuracy": report["accuracy_micro"]})
with comp_col:
    st.markdown("#### Accuracy by complexity")
    if report["acc_by_complexity"]:
        bar_chart(report["acc_by_complexity"])
        st.caption("Partial matching: t_k = 1 for 1-2 intentions, t_k = 2 for 3.")
    else:
        st.caption("Not enough multi-intention samples yet.")

st.markdown("#### Set-level agreement with the reference")
agreement = outcome["agreement"]
agree_cols = st.columns(4)
agree_cols[0].metric("Exact match", f"{agreement['exact_match']:.1%}")
agree_cols[1].metric("Partial match", f"{agreement['partial_match']:.1%}")
agree_cols[2].metric("No overlap", f"{agreement['no_overlap']:.1%}")
agree_cols[3].metric("Mean Jaccard", f"{agreement['mean_jaccard']:.4f}")

# ---------------------------------------------------------------------------
st.markdown("### Per-class metrics")
st.dataframe([{"Intention": category,
               "Support": report["support"].get(category, 0),
               "Precision": values["precision"], "Recall": values["recall"],
               "F1": values["f1"], "Accuracy": values["accuracy"],
               "TP": values["tp"], "FP": values["fp"],
               "FN": values["fn"], "TN": values["tn"]}
              for category, values in report["per_class"].items()],
             hide_index=True, **FULL_WIDTH)

# ---------------------------------------------------------------------------
st.markdown("### Credential-theft benchmark")
st.caption("Single-category comparison, matching the PhishIntention benchmark "
           "setup of the base paper.")
benchmark = outcome["credential_theft_benchmark"]
bench_cols = st.columns(4)
bench_cols[0].metric("Precision", f"{benchmark['precision']:.4f}")
bench_cols[1].metric("Recall", f"{benchmark['recall']:.4f}")
bench_cols[2].metric("F1", f"{benchmark['f1']:.4f}")
bench_cols[3].metric("Accuracy", f"{benchmark['accuracy']:.4f}")

# ---------------------------------------------------------------------------
st.markdown("### Error analysis")
rows = evaluator.per_sample(run_path, reference)
outcome_filter = st.radio("Show", ["all", "exact", "partial", "miss"],
                          horizontal=True)
visible = [r for r in rows if outcome_filter == "all"
           or r["outcome"] == outcome_filter]
st.caption(f"{len(visible)} of {len(rows)} sample(s).")
st.dataframe(visible[:400], hide_index=True, **FULL_WIDTH)

# ---------------------------------------------------------------------------
st.markdown("### Intention combinations")
combo_cols = st.columns(2)
with combo_cols[0]:
    st.caption("Predicted by the framework")
    bar_chart(outcome["co_occurrence_predicted"])
with combo_cols[1]:
    st.caption(f"In the {outcome['reference']}")
    bar_chart(outcome["co_occurrence_reference"])

st.markdown("### Sector × intention matrix")
matrix = outcome.get("sector_matrix") or {}
if matrix:
    st.dataframe([{"Sector": s, **c} for s, c in matrix.items()],
                 hide_index=True, **FULL_WIDTH)

# ---------------------------------------------------------------------------
st.markdown("### Reference quality")
st.caption("How trustworthy is the manifest itself? Low inter-annotator "
           "agreement or a large review backlog weakens every number above.")
quality = evaluator.manifest_quality()
if quality.get("total"):
    q_cols = st.columns(5)
    q_cols[0].metric("Manifest records", quality["total"])
    q_cols[1].metric("Inter-annotator agreement",
                     f"{quality['inter_annotator_agreement']:.1%}")
    q_cols[2].metric("Tie-breaks", quality["tie_breaks"])
    q_cols[3].metric("Needs review", quality["needs_review"])
    q_cols[4].metric("Human verified", quality["human_verified"])
    qa, qb = st.columns(2)
    with qa:
        st.caption("Agreement breakdown")
        bar_chart(quality["agreement_breakdown"])
    with qb:
        st.caption("Reference label distribution")
        bar_chart(quality["per_category"])

# ---------------------------------------------------------------------------
st.markdown("### Framework vs single-agent baseline")
if len(runs) >= 2:
    compare_cols = st.columns(2)
    framework_run = compare_cols[0].selectbox("Multi-agent run",
                                              [p.name for p in runs], key="fw")
    baseline_run = compare_cols[1].selectbox("Single-agent run",
                                             [p.name for p in runs], key="bl")
    if st.button("Compare runs", type="primary"):
        comparison = evaluator.compare_runs(
            str(next(p for p in runs if p.name == framework_run)),
            str(next(p for p in runs if p.name == baseline_run)), reference)
        table = []
        for name, payload in comparison.items():
            if "error" in payload:
                st.warning(f"{name}: {payload['error']}")
                continue
            block = payload["report"]
            table.append({"Framework": name,
                          "Precision_micro": block["precision_micro"],
                          "Recall_micro": block["recall_micro"],
                          "F1_micro": block["f1_micro"],
                          "Accuracy_micro": block["accuracy_micro"],
                          "Overall Accuracy": block["overall_accuracy"]})
        if table:
            st.dataframe(table, hide_index=True, **FULL_WIDTH)
            if len(table) == 2 and table[1]["Precision_micro"]:
                gain = ((table[0]["Precision_micro"] - table[1]["Precision_micro"])
                        / table[1]["Precision_micro"] * 100)
                st.success(f"Multi-agent precision gain over single-agent: "
                           f"**{gain:+.1f}%**")
else:
    st.caption("Save at least two runs (one per pipeline) to compare them here.")

with st.expander("🧾 Raw evaluation payload"):
    st.json({k: v for k, v in outcome.items() if k != "sample_ids"})

footer()
