"""Shared Streamlit styling and widgets for the PhishIntentionLLM console."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Make `src/` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phishintentionllm.schemas import (AgentStep, AnalysisResult,  # noqa: E402
                                       Intention, IntentionResult)

# ---------------------------------------------------------------------------
# Streamlit version compatibility
#   `use_container_width=True` was renamed to `width="stretch"`.
# ---------------------------------------------------------------------------
def _detect_width_kwarg() -> Dict[str, Any]:
    """Prefer the modern `width="stretch"` API, fall back on older releases."""
    try:
        params = inspect.signature(st.dataframe).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {"use_container_width": True}


FULL_WIDTH: Dict[str, Any] = _detect_width_kwarg()


def bar_chart(data, horizontal: bool = True, height: Optional[int] = None) -> None:
    """Version-tolerant bar chart."""
    if not data:
        st.caption("No data to plot.")
        return
    try:
        st.bar_chart(data, horizontal=horizontal, height=height)
    except TypeError:
        import pandas as pd
        frame = (pd.Series(data, name="value").to_frame()
                 if isinstance(data, dict) else data)
        st.bar_chart(frame, height=height)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
CATEGORY_STYLE: Dict[str, Dict[str, str]] = {
    "Credential Theft": {"color": "#ff4b6e", "icon": "🔑", "short": "CT",
                         "plain": "Steals usernames & passwords"},
    "Financial Fraud": {"color": "#ffb020", "icon": "💳", "short": "FF",
                        "plain": "Targets your money & card details"},
    "Malware Distribution": {"color": "#a855f7", "icon": "🦠", "short": "MD",
                             "plain": "Pushes malicious software"},
    "Personal Information Harvesting": {"color": "#22d3ee", "icon": "🪪",
                                        "short": "PIH",
                                        "plain": "Collects your identity data"},
}

LAYER_ICON = {
    "Layer 0 - Input": "🖼️",
    "Layer 1 - Perception": "👁️",
    "Layer 2 - Semantic": "🧠",
    "Layer 3 - Classification": "🏷️",
    "Layer 4 - Specialist Analysis": "🔬",
    "Layer 5 - Validation": "✅",
    "Manifest - Input": "🖼️",
    "Manifest - Independent Annotation": "🗳️",
    "Manifest - Tie-Break": "⚖️",
    "Manifest - Finalisation": "📝",
    "Baseline": "➖",
}

STATUS_STYLE = {
    "pending": ("#64748b", "○", "Queued"),
    "running": ("#38bdf8", "◐", "Running"),
    "done": ("#22c55e", "●", "Done"),
    "skipped": ("#64748b", "◌", "Skipped"),
    "error": ("#ef4444", "✕", "Failed"),
}

STAGE_STYLE = {
    "manifest": ("#f59e0b", "Manifest stage"),
    "framework": ("#38bdf8", "Framework stage"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.pil-hero {
  background: linear-gradient(120deg, #0b1220 0%, #16213e 45%, #3a1c5c 100%);
  border: 1px solid rgba(148,163,184,.22);
  border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
  position: relative; overflow: hidden;
}
.pil-hero::after {
  content:''; position:absolute; right:-70px; top:-70px; width:260px; height:260px;
  background: radial-gradient(circle, rgba(56,189,248,.28), transparent 65%);
}
.pil-hero h1 {
  margin:0; font-size: 2.35rem; font-weight: 800; letter-spacing:-.5px;
  background: linear-gradient(90deg,#38bdf8,#a855f7 55%,#ff4b6e);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pil-hero p { margin:.45rem 0 0; color:#cbd5e1; font-size:1.02rem; }
.pil-hero .pil-tags { margin-top:14px; display:flex; gap:8px; flex-wrap:wrap; }
.pil-tag {
  font-size:.72rem; font-weight:600; letter-spacing:.4px; text-transform:uppercase;
  padding:4px 11px; border-radius:999px; color:#e2e8f0;
  background:rgba(148,163,184,.14); border:1px solid rgba(148,163,184,.3);
}

.pil-card {
  background: rgba(15,23,42,.62); border:1px solid rgba(148,163,184,.2);
  border-radius:16px; padding:16px 18px; margin-bottom:12px;
}
.pil-card h4 { margin:0 0 8px; font-size:.95rem; color:#e2e8f0; letter-spacing:.2px; }

/* --- stage banner inside the flow --- */
.pil-stage {
  display:flex; align-items:center; gap:9px; margin:10px 0 7px;
  font-size:.72rem; font-weight:700; letter-spacing:1px; text-transform:uppercase;
  color:var(--stage); }
.pil-stage::after {
  content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--stage),transparent); }

/* --- agent flow --- */
.pil-flow { display:flex; flex-direction:column; gap:7px; }
.pil-node {
  display:flex; align-items:flex-start; gap:12px; padding:11px 14px;
  border-radius:13px; border:1px solid rgba(148,163,184,.2);
  background: rgba(15,23,42,.55); transition: all .2s ease;
}
.pil-node.running { border-color:#38bdf8;
  box-shadow:0 0 0 1px rgba(56,189,248,.35), 0 0 22px rgba(56,189,248,.18); }
.pil-node.done { border-color: rgba(34,197,94,.45); }
.pil-node.error { border-color: rgba(239,68,68,.55); }
.pil-node.skipped { opacity:.5; }
.pil-node .pil-ico { font-size:1.25rem; line-height:1.5rem; width:26px; text-align:center; }
.pil-node .pil-body { flex:1; min-width:0; }
.pil-node .pil-title { font-weight:600; color:#f1f5f9; font-size:.93rem;
  display:flex; align-items:center; gap:7px; }
.pil-vlm {
  font-size:.6rem; font-weight:700; letter-spacing:.6px; padding:2px 6px;
  border-radius:5px; background:rgba(34,211,238,.16); color:#22d3ee;
  border:1px solid rgba(34,211,238,.4); }
.pil-node .pil-meta {
  font-family:'JetBrains Mono',monospace; font-size:.71rem; color:#94a3b8; margin-top:3px; }
.pil-node .pil-summary { color:#cbd5e1; font-size:.84rem; margin-top:5px; }
.pil-node .pil-status { font-size:.7rem; font-weight:700; text-transform:uppercase;
  letter-spacing:.6px; white-space:nowrap; padding-top:3px; }
.pil-connector { color:#475569; text-align:center; font-size:.85rem;
  line-height:.6rem; margin:-3px 0; }

/* --- intention cards --- */
.pil-intent {
  border-radius:15px; padding:15px 17px; margin-bottom:11px;
  border:1px solid var(--accent);
  background:linear-gradient(135deg, var(--wash), rgba(15,23,42,.6));
}
.pil-intent .pil-head { display:flex; align-items:center; gap:10px; }
.pil-intent .pil-name { font-weight:700; font-size:1.06rem; color:#f8fafc; }
.pil-intent .pil-chip {
  margin-left:auto; font-family:'JetBrains Mono',monospace; font-weight:600;
  font-size:.82rem; padding:3px 10px; border-radius:999px;
  background:var(--accent); color:#0b1220; }
.pil-intent .pil-plain { color:#e2e8f0; font-size:.88rem; margin:7px 0 9px; }
.pil-bar { height:8px; border-radius:999px; background:rgba(148,163,184,.2); overflow:hidden; }
.pil-bar > div { height:100%; border-radius:999px; background:var(--accent); }
.pil-ev { margin:9px 0 0; padding-left:18px; color:#cbd5e1; font-size:.83rem; }
.pil-ev li { margin-bottom:3px; }

/* --- verdict banner --- */
.pil-verdict { border-radius:17px; padding:19px 22px; margin-bottom:14px;
  border:1px solid var(--accent);
  background:linear-gradient(120deg, var(--wash), rgba(15,23,42,.72)); }
.pil-verdict .pil-label { font-size:.73rem; letter-spacing:1.2px;
  text-transform:uppercase; color:#94a3b8; font-weight:700; }
.pil-verdict .pil-line { font-size:1.4rem; font-weight:800; color:#f8fafc; margin:4px 0 8px; }
.pil-verdict .pil-sub { color:#cbd5e1; font-size:.92rem; }

.pil-kv { display:flex; gap:7px; flex-wrap:wrap; margin-top:10px; }
.pil-kv span {
  font-size:.74rem; padding:4px 10px; border-radius:8px; color:#e2e8f0;
  background:rgba(148,163,184,.14); border:1px solid rgba(148,163,184,.26);
  font-family:'JetBrains Mono',monospace; }
.pil-foot { color:#64748b; font-size:.78rem; text-align:center; margin-top:26px;
  padding-top:12px; border-top:1px solid rgba(148,163,184,.16); }
</style>
"""


def style_page() -> None:
    """Inject the shared CSS. Safe to call from any view.

    Unlike `setup_page`, this does NOT call `st.set_page_config` - that belongs
    to the navigation entry point only, and calling it twice raises.
    """
    st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
def setup_page(title: str, icon: str = "🎣", layout: str = "wide") -> None:
    st.set_page_config(page_title=f"{title} | PhishIntentionLLM", page_icon=icon,
                       layout=layout, initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, tags: Optional[List[str]] = None) -> None:
    tag_html = "".join(f"<span class='pil-tag'>{t}</span>" for t in (tags or []))
    st.markdown(f"""<div class="pil-hero"><h1>{title}</h1><p>{subtitle}</p>
        <div class="pil-tags">{tag_html}</div></div>""", unsafe_allow_html=True)


def category_style(category: str) -> Dict[str, str]:
    return CATEGORY_STYLE.get(category, {"color": "#94a3b8", "icon": "❔",
                                         "short": "?", "plain": ""})


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Agent flow
# ---------------------------------------------------------------------------
def render_agent_flow(steps: List[AgentStep], compact: bool = False,
                      title: str = "Multi-Agent Flow") -> str:
    """Render the live agent pipeline as connected, stage-grouped nodes."""
    if not steps:
        return (f"<div class='pil-card'><h4>🔄 {title}</h4>"
                "<div style='color:#94a3b8;font-size:.86rem'>"
                "Waiting for the pipeline to start…</div></div>")

    nodes: List[str] = []
    current_stage: Optional[str] = None

    for index, step in enumerate(steps):
        data = step.to_dict() if isinstance(step, AgentStep) else dict(step)
        stage = data.get("stage", "framework")
        if stage != current_stage:
            colour, label = STAGE_STYLE.get(stage, ("#94a3b8", stage))
            if current_stage is not None:
                nodes.append("<div class='pil-connector'>│<br>▼</div>")
            nodes.append(f"<div class='pil-stage' style='--stage:{colour}'>{label}</div>")
            current_stage = stage
        elif index > 0:
            nodes.append("<div class='pil-connector'>│<br>▼</div>")

        status = data.get("status", "pending")
        colour, glyph, label = STATUS_STYLE.get(status, STATUS_STYLE["pending"])
        icon = LAYER_ICON.get(str(data.get("layer", "")).split(" (")[0], "🤖")
        duration = data.get("duration", 0) or 0
        summary = data.get("error") or data.get("summary") or ""
        if compact and len(summary) > 110:
            summary = summary[:110] + "…"
        vlm_badge = "<span class='pil-vlm'>VLM · sees image</span>" \
            if data.get("vision") else ""

        nodes.append(
            f"""<div class="pil-node {status}">
  <div class="pil-ico">{icon}</div>
  <div class="pil-body">
    <div class="pil-title">{data.get('agent', 'Agent')} {vlm_badge}</div>
    <div class="pil-meta">{data.get('layer', '')} · {data.get('model', '')}"""
            + (f" · {duration}s" if duration else "")
            + f"""</div>
    {f"<div class='pil-summary'>{summary}</div>" if summary else ""}
  </div>
  <div class="pil-status" style="color:{colour}">{glyph} {label}</div>
</div>""")

    return (f"<div class='pil-card'><h4>🔄 {title}</h4>"
            "<div class='pil-flow'>" + "".join(nodes) + "</div></div>")


def flow_legend(stage: str = "framework") -> None:
    if stage == "manifest":
        st.caption("🗳️ Annotator A + Annotator B (independent VLM labels) → "
                   "⚖️ Tie-breaker (only on dispute) → 📝 Finaliser → manifest row. "
                   "Every agent receives the screenshot.")
    else:
        st.caption("👁️ Perception → 🧠 Context Enrichment → 🏷️ Classification → "
                   "🔬 Four expert agents → ✅ Validation & Synthesis. "
                   "Every layer analyses the image with a VLM.")


# ---------------------------------------------------------------------------
# Result rendering
# ---------------------------------------------------------------------------
def render_verdict(result: AnalysisResult) -> None:
    accent = "#ff4b6e" if result.is_phishing else "#22c55e"
    wash = _rgba(accent, 0.16)
    label = "PHISHING DETECTED" if result.is_phishing else "NO PHISHING INTENT DETECTED"

    chips = [f"phishing score {result.phishing_score:.0%}",
             f"confidence {result.overall_confidence:.0%}",
             f"sector: {result.sector}",
             f"{result.vision_calls} VLM calls",
             f"{result.elapsed:.1f}s"]
    if result.feedback_loop_used:
        chips.append("feedback loop")

    st.markdown(f"""<div class="pil-verdict" style="--accent:{accent};--wash:{wash}">
  <div class="pil-label">{label}</div>
  <div class="pil-line">{result.verdict or 'Phishing website'}</div>
  <div class="pil-sub">{result.risk_summary or ''}</div>
  <div class="pil-kv">{''.join(f'<span>{c}</span>' for c in chips)}</div>
</div>""", unsafe_allow_html=True)


def render_intentions(intentions: List[IntentionResult]) -> None:
    if not intentions:
        st.info("No intention was established for this screenshot.")
        return
    for intention in intentions:
        style = category_style(intention.category)
        accent = style["color"]
        evidence = "".join(f"<li>{e}</li>" for e in intention.evidence[:5])
        st.markdown(
            f"""<div class="pil-intent" style="--accent:{accent};--wash:{_rgba(accent, .14)}">
  <div class="pil-head">
    <span style="font-size:1.35rem">{style['icon']}</span>
    <span class="pil-name">{intention.category}</span>
    <span class="pil-chip">{intention.confidence:.0%} · {intention.risk_band}</span>
  </div>
  <div class="pil-plain">{intention.explanation or style['plain']}</div>
  <div class="pil-bar"><div style="width:{max(3, min(100, intention.confidence*100)):.0f}%"></div></div>
  {f"<ul class='pil-ev'>{evidence}</ul>" if evidence else ""}
</div>""", unsafe_allow_html=True)


def render_label_chips(labels: List[str], title: str = "") -> None:
    if not labels:
        st.caption(f"{title}—" if title else "—")
        return
    chips = "".join(
        f"<span style=\"font-size:.78rem;padding:4px 11px;border-radius:999px;"
        f"margin-right:6px;background:{_rgba(category_style(c)['color'], .18)};"
        f"color:{category_style(c)['color']};border:1px solid "
        f"{_rgba(category_style(c)['color'], .45)}\">"
        f"{category_style(c)['icon']} {c}</span>" for c in labels)
    st.markdown((f"<div style='margin:4px 0'><span style='color:#94a3b8;"
                 f"font-size:.78rem;margin-right:8px'>{title}</span>{chips}</div>"
                 if title else f"<div style='margin:4px 0'>{chips}</div>"),
                unsafe_allow_html=True)


def render_votes(votes) -> None:
    """Annotator vote cards (manifest stage)."""
    if not votes:
        return
    columns = st.columns(len(votes))
    for column, vote in zip(columns, votes):
        data = vote if isinstance(vote, dict) else vote.to_dict()
        with column:
            if data.get("error"):
                st.markdown(f"""<div class="pil-card"><h4>🗳️ {data['annotator']}</h4>
<div style="color:#ef4444;font-size:.83rem">Failed: {data['error'][:160]}</div></div>""",
                            unsafe_allow_html=True)
                continue
            confidences = data.get("confidences", {})
            labels = ", ".join(f"{Intention.short(c)} {confidences.get(c, 0):.0%}"
                               for c in data.get("categories", [])) or "no intention"
            st.markdown(f"""<div class="pil-card">
  <h4>🗳️ {data.get('annotator', 'Annotator')}</h4>
  <div style="font-family:'JetBrains Mono',monospace;font-size:.74rem;color:#94a3b8">
    {data.get('model', '')} · {data.get('latency', 0):.1f}s</div>
  <div style="margin-top:8px;color:#f1f5f9;font-weight:600">{labels}</div>
  <div style="margin-top:7px;color:#cbd5e1;font-size:.83rem">
    {str(data.get('rationale', ''))[:280]}</div>
</div>""", unsafe_allow_html=True)


def render_specialists(result: AnalysisResult) -> None:
    if not result.specialist_findings:
        st.caption("No specialist was activated.")
        return
    rows = []
    for finding in result.specialist_findings:
        style = category_style(finding.category)
        rows.append({"Expert": f"{style['icon']} {finding.category}",
                     "Verdict": "✅ Confirmed" if finding.confirmed else "❌ Rejected",
                     "Confidence": f"{finding.confidence:.0%}",
                     "Key evidence": "; ".join(finding.evidence[:2]) or "—"})
    st.dataframe(rows, hide_index=True, **FULL_WIDTH)


def result_summary_row(result: AnalysisResult) -> Dict[str, Any]:
    return {"Sample": result.sample_id.split("::")[-1][:28],
            "Source": result.source,
            "Intentions": ", ".join(i.short for i in result.intentions) or "—",
            "Confidence": f"{result.overall_confidence:.0%}",
            "Sector": result.sector,
            "VLM calls": result.vision_calls,
            "Time": f"{result.elapsed:.1f}s"}


def manifest_summary_row(record) -> Dict[str, Any]:
    data = record if isinstance(record, dict) else record.to_dict()
    return {"Sample": data["sample_id"].split("::")[-1][:28],
            "Source": data.get("source", ""),
            "Labels": ", ".join(Intention.short(c)
                                for c in data.get("labels", [])) or "—",
            "Agreement": data.get("agreement", "—"),
            "Tie-break": "yes" if data.get("tie_break_used") else "no",
            "Review": "⚠️" if data.get("needs_review") else "✓",
            "Time": f"{data.get('elapsed', 0):.1f}s"}


def footer() -> None:
    st.markdown("<div class='pil-foot'>PhishIntentionLLM · Multi-Agent "
                "Retrieval-Augmented Generation · every agent is a local "
                "open-source vision-language model served by Ollama</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_config():
    from phishintentionllm.config import load_config
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


@st.cache_resource(show_spinner="Starting the multi-agent framework…")
def get_framework():
    from phishintentionllm.pipeline import PhishIntentionLLM
    return PhishIntentionLLM(get_config())


@st.cache_resource(show_spinner="Starting the annotator ensemble…")
def get_manifest_builder():
    from phishintentionllm.annotation import ManifestBuilder
    return ManifestBuilder(get_config(), client=get_framework().client,
                           store=get_store())


@st.cache_resource(show_spinner=False)
def get_registry():
    from phishintentionllm.datasets import DatasetRegistry
    return DatasetRegistry(get_config())


@st.cache_resource(show_spinner=False)
def get_store():
    from phishintentionllm.annotation import AnnotationStore
    return AnnotationStore(get_config())


@st.cache_resource(show_spinner=False)
def get_manual_manager():
    from phishintentionllm.annotation import ManualAnnotationManager
    return ManualAnnotationManager(get_config(), get_store())


@st.cache_resource(show_spinner=False)
def get_evaluator():
    from phishintentionllm.evaluation import Evaluator
    return Evaluator(get_config(), get_store())


def sidebar_status() -> None:
    """Ollama connectivity, model readiness and manifest coverage."""
    cfg = get_config()
    with st.sidebar:
        st.markdown("### ⚙️ Runtime")
        try:
            report = get_framework().preflight()
        except Exception as exc:
            st.error(f"Framework unavailable: {exc}")
            return

        if report.get("alive"):
            st.success(f"Ollama online · {report['host']}")
        else:
            st.error(f"Ollama offline · {report['host']}")
            st.caption("Start it with `ollama serve`, then reload this page.")

        manifest_roles = {r: i for r, i in report.get("models", {}).items()
                          if i["stage"] == "manifest"}
        framework_roles = {r: i for r, i in report.get("models", {}).items()
                           if i["stage"] == "framework"}

        st.markdown("**Manifest stage** (annotators)")
        for role, info in manifest_roles.items():
            st.caption(f"{'✅' if info['ready'] else '⬇️'} {role} — `{info['model']}`")
        st.markdown("**Framework stage** (5 layers)")
        for role, info in framework_roles.items():
            st.caption(f"{'✅' if info['ready'] else '⬇️'} {role} — `{info['model']}`")

        if report.get("missing"):
            st.warning("Missing models — pull them:")
            st.code("\n".join(f"ollama pull {m}" for m in report["missing"]),
                    language="bash")
        if report.get("non_vision"):
            st.error("Not multimodal — every agent needs a VLM:\n"
                     + ", ".join(report["non_vision"]))

        st.divider()
        store = get_store()
        st.caption(f"📋 Manifest rows: **{len(store.manifest_ids())}**")
        st.caption(f"🎯 Predictions: **{len(store.load_predictions())}**")
        st.caption(f"τ = {cfg.get('framework.confidence_threshold')} · "
                   f"reference = `{cfg.get('evaluation.reference')}`")
