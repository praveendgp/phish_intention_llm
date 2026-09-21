"""Browse both dataset sources side by side despite their different layouts."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (FULL_WIDTH, bar_chart, footer, get_registry, get_store,
                           hero, setup_page, sidebar_status)

setup_page("Dataset Explorer", "📚")
registry = get_registry()
store = get_store()

hero("Dataset Explorer",
     "Two phishing corpora with completely different folder structures, "
     "normalised into a single sample schema.",
     tags=["Putra (record folders)", "Phish-IRIS (brand folders)",
           "Unified Sample schema"])
sidebar_status()

st.markdown("### Source status")
cols = st.columns(len(registry.available()))
for column, info in zip(cols, registry.available()):
    with column:
        icon = "✅" if info["exists"] else "❌"
        st.markdown(
            f"""<div class="pil-card">
<h4>{icon} {info['source']}</h4>
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#94a3b8;
word-break:break-all">{info['path']}</div>
<div style="margin-top:6px;color:#cbd5e1;font-size:.83rem">
{'Ready' if info['exists'] else 'Folder not found'} ·
{'enabled' if info['enabled'] else 'disabled in config'}</div></div>""",
            unsafe_allow_html=True)

st.caption("Expected layout — **Putra**: "
           "`putra/<phishing|not-phishing>/<record-id>/screenshots/*.png` · "
           "**Phish-IRIS**: `phishIris/<train|val>/<brand>/*.png`")

existing = [d["source"] for d in registry.available() if d["exists"]]
if not existing:
    st.warning("Place at least one dataset under `data/raw/` to explore it here.")
    footer()
    st.stop()

st.markdown("### Browse samples")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    source = st.selectbox("Source", existing)
with col_b:
    limit = st.number_input("Load first N", 20, 20000, 300, step=20)
with col_c:
    include_benign = st.checkbox("Include benign (Putra)", value=False)
with col_d:
    per_page = st.selectbox("Thumbnails / page", [12, 24, 48], index=1)

samples = registry.load_source(source, limit=int(limit),
                               phishing_only=not include_benign)
if not samples:
    st.warning("No samples matched.")
    footer()
    st.stop()

brands = Counter(s.brand for s in samples if s.brand)
splits = Counter(s.split for s in samples if s.split)
metrics = st.columns(4)
metrics[0].metric("Samples loaded", len(samples))
metrics[1].metric("Phishing", sum(1 for s in samples if s.is_phishing))
metrics[2].metric("Distinct brands", len(brands) or "—")
metrics[3].metric("Splits / labels", len(splits) or "—")

if brands:
    with st.expander("Brand distribution"):
        bar_chart(dict(brands.most_common(20)))

filters = st.columns([1, 2])
with filters[0]:
    brand_filter = st.selectbox("Filter by brand", ["(all)"] + sorted(brands))
with filters[1]:
    search = st.text_input("Search sample id / path")

filtered = [s for s in samples
            if (brand_filter == "(all)" or s.brand == brand_filter)
            and (not search or search.lower() in s.sample_id.lower()
                 or search.lower() in s.screenshot_path.lower())]
st.caption(f"{len(filtered)} sample(s) after filtering.")

pages = max(1, (len(filtered) + per_page - 1) // per_page)
page = st.number_input("Page", 1, pages, 1)
window = filtered[(page - 1) * per_page: page * per_page]

manifest = store.manifest_map()
grid = st.columns(4)
for index, sample in enumerate(window):
    with grid[index % 4]:
        try:
            st.image(sample.screenshot_path, **FULL_WIDTH)
        except Exception:
            st.caption("⚠️ unreadable image")
        short = sample.sample_id.split("::")[-1]
        badge = "🎣" if sample.is_phishing else "🟢"
        labels = manifest.get(sample.sample_id)
        st.caption(f"{badge} `{short[:22]}`"
                   + (f" · **{sample.brand}**" if sample.brand else "")
                   + (f"<br>📋 {', '.join(labels)}" if labels else
                      "<br><span style='color:#64748b'>not in manifest</span>"),
                   unsafe_allow_html=True)
        if st.button("Analyse", key=f"an_{sample.sample_id}", **FULL_WIDTH):
            st.session_state["preselect_sample"] = sample.to_dict()
            st.switch_page("streamlit_app.py")

with st.expander("📋 Raw sample records"):
    st.dataframe([{"sample_id": s.sample_id, "source": s.source,
                   "brand": s.brand or "—", "split": s.split or "—",
                   "phishing": s.is_phishing, "url": s.url or "—",
                   "screenshot": s.screenshot_path} for s in window],
                 hide_index=True, **FULL_WIDTH)

footer()
