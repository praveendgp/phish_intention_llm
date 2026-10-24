"""SSO Callback Handler to ensure Clerk redirects never 404."""
import streamlit as st
from phishintentionllm.auth_clerk import check_realtime_clerk_callback

check_realtime_clerk_callback()
st.success("Authentication received! Redirecting to Intent Scanner...")
st.switch_page("views/intent_scanner.py")
