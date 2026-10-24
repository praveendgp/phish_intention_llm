"""SSO Callback Handler to ensure Clerk redirects never 404."""
import streamlit as st
from phishintentionllm.config import load_config
from phishintentionllm.auth_clerk import check_realtime_clerk_callback

cfg = load_config("config.yaml")
check_realtime_clerk_callback(cfg)

if st.session_state.get("authenticated", False):
    st.success("Authentication confirmed! Redirecting to Intent Scanner...")
    st.switch_page("views/intent_scanner.py")
else:
    st.info("Syncing session with Clerk...")
    st.rerun()
