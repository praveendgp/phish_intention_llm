"""
PhishIntentionLLM - Main Streamlit Entry Point with Real-Time Clerk OAuth & Subscription RBAC.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "app"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from phishintentionllm.config import load_config
from phishintentionllm.auth_clerk import (
    init_clerk_auth, 
    render_clerk_auth_sidebar, 
    check_realtime_clerk_callback,
    render_clerk_login_gate
)

st.set_page_config(
    page_title="PhishIntentionLLM | Threat Intelligence",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration using the supported load_config() function
cfg = load_config("config.yaml")

# 1. Initialize auth & check for incoming Clerk OAuth redirect callbacks
init_clerk_auth(cfg)
check_realtime_clerk_callback(cfg)

# 2. Render sidebar login / user profile / subscription status
render_clerk_auth_sidebar(cfg)

# 3. GATE: If user is not authenticated with Google/GitHub, show login gate and STOP!
if not st.session_state.get("authenticated", False):
    render_clerk_login_gate(cfg)
    st.stop()

# 4. Authenticated navigation for logged in users
VIEWS = Path(__file__).parent / "views"
pages = [
    st.Page(str(VIEWS / "intent_scanner.py"), title="Intent Scanner", icon="🎣", default=True),
    st.Page(str(VIEWS / "manual_annotation.py"), title="Threat Intelligence Archive", icon="🗄️"),
    st.Page(str(VIEWS / "knowledge_base.py"), title="RAG Knowledge Base", icon="🧠"),
]

nav = st.navigation(pages)
nav.run()
