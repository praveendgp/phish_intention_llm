"""PhishIntentionLLM - console entry point and navigation.

Declaring the pages with `st.navigation` lets the landing page carry its own
title and icon in the sidebar, instead of Streamlit falling back to the script
filename ("streamlit app"). It also disables the automatic `pages/` discovery,
so only the three views below are exposed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src/` and `app/` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "app"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

st.set_page_config(
    page_title="PhishIntentionLLM",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

VIEWS = Path(__file__).parent / "views"

pages = [
    st.Page(str(VIEWS / "intent_scanner.py"),
            title="Intent Scanner", icon="🎣", default=True),
    st.Page(str(VIEWS / "manual_annotation.py"),
            title="Manual Annotation", icon="✍️"),
    st.Page(str(VIEWS / "knowledge_base.py"),
            title="Knowledge Base", icon="🧠"),
]

st.navigation(pages).run()
