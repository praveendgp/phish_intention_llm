"""Inspect and query the dual-layer RAG knowledge architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_components import (category_style, footer, get_config, hero, setup_page,
                           sidebar_status)
from phishintentionllm.rag.knowledge_base import get_knowledge_base
from phishintentionllm.rag.retriever import KnowledgeRetriever

setup_page("Knowledge Base", "🧠")
cfg = get_config()
kb = get_knowledge_base()
retriever = KnowledgeRetriever(cfg, kb)

hero("Knowledge Base",
     "The dual-layer repository that grounds every agent: domain-agnostic threat "
     "patterns K_B and category-specific expert knowledge K_c.",
     tags=["K_B = {P_c, P_v, P_t}", "K_c = {F_c, D_c}",
           "Used by annotators and framework", f"backend: {retriever.backend}"])
sidebar_status()

stats = kb.stats()
cols = st.columns(4)
cols[0].metric("Total passages", stats["passages"])
cols[1].metric("Basic repository", stats["basic_passages"])
cols[2].metric("Specialist repository", stats["specialist_passages"])
cols[3].metric("Categories", stats["categories"])

st.markdown("### 🔎 Try a retrieval query")
query = st.text_input("Describe what a screenshot shows",
                      "login form asking for password and one-time code")
filter_col, topk_col = st.columns(2)
category_filter = filter_col.selectbox("Restrict to category",
                                       ["(all)"] + cfg.categories)
top_k = topk_col.slider("Passages to retrieve", 1, 12, 5)

if query:
    hits = retriever.retrieve_with_scores(
        query, top_k, None if category_filter == "(all)" else category_filter)
    if not hits:
        st.caption("No matching passage.")
    for passage, score in hits:
        scope = passage.category or "general"
        icon = category_style(passage.category)["icon"] if passage.category else "📘"
        st.markdown(f"""<div class="pil-card">
<div style="display:flex;gap:9px;align-items:center">
  <span style="font-size:1.1rem">{icon}</span>
  <span style="font-weight:600;color:#f1f5f9">{scope}</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#94a3b8">
    {passage.facet}</span>
  <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;
    font-size:.74rem;color:#38bdf8">sim {score}</span>
</div>
<div style="margin-top:7px;color:#cbd5e1;font-size:.88rem">{passage.text}</div>
</div>""", unsafe_allow_html=True)

st.markdown("### 📗 Basic Threat Pattern Repository (K_B)")
for title, key in {"P_c · Common phishing patterns": "common_patterns",
                   "P_v · Visual deception techniques": "visual_deception",
                   "P_t · Text manipulation patterns": "text_manipulation",
                   "P_u · URL red flags": "url_red_flags"}.items():
    with st.expander(f"{title} ({len(kb.basic.get(key, []))})"):
        for item in kb.basic.get(key, []):
            st.markdown(f"- {item}")

st.markdown("### 📕 Specialist Knowledge Repository (K_c)")
for category in cfg.categories:
    knowledge = kb.retrieve_specialist_knowledge(category)
    style = category_style(category)
    with st.expander(f"{style['icon']} {category}"):
        st.caption(knowledge.get("definition", ""))
        for label, key in {"F_c · Primary features": "F_c_primary_features",
                           "T_c · Common targets": "T_c_common_targets",
                           "M_c · Specialised techniques": "M_c_techniques",
                           "I_c · Distinctive indicators": "I_c_indicators",
                           "Counter-indicators": "counter_indicators"}.items():
            st.markdown(f"**{label}**")
            for item in knowledge.get(key, []):
                st.markdown(f"- {item}")

st.markdown("### 🔗 Co-occurrence priors")
for note in kb.co_occurrence_notes():
    st.markdown(f"- {note}")

st.info("Edit `data/knowledge_base/*.json` to extend the knowledge base — both "
        "the annotator ensemble and the framework pick up changes on the next run.")
footer()
