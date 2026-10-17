"""Subscription and authentication management for PhishIntentionLLM."""

from __future__ import annotations

import streamlit as st
from typing import Dict, Any


def init_auth(cfg) -> None:
    """Initialize session state authentication variables."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["plan"] = "free"


def current_plan(cfg) -> Dict[str, Any]:
    """Return the configuration dictionary for the active plan."""
    init_auth(cfg)
    plan_key = st.session_state.get("plan", "free")
    plans = cfg.get("subscription.plans", {})
    return plans.get(plan_key, plans.get("free", {}))


def login(username: str, password: str, cfg) -> bool:
    """Validate credentials against config accounts."""
    accounts = cfg.get("subscription.accounts", {})
    account = accounts.get(username)
    if account and str(account.get("password")) == str(password):
        st.session_state["authenticated"] = True
        st.session_state["user"] = username
        st.session_state["plan"] = account.get("plan", "free")
        return True
    return False


def logout() -> None:
    """Reset session state to free guest status."""
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state["plan"] = "free"
    st.session_state.pop("last_result", None)


def render_auth_widget(cfg) -> None:
    """Render the subscription login/status panel at the VERY TOP of the sidebar."""
    init_auth(cfg)
    plan_info = current_plan(cfg)

    with st.sidebar:
        st.markdown("### 👤 Access & Subscription")
        if st.session_state["authenticated"]:
            badge = "💎 **PRO**" if st.session_state["plan"] == "pro" else "🆓 **FREE**"
            st.success(f"{badge} logged in as `{st.session_state['user']}`")
            if st.button("🚪 Sign Out", use_container_width=True):
                logout()
                st.rerun()
        else:
            st.info("Current Tier: 🆓 **Free Community**")
            with st.expander("🔑 **Sign In to Pro**", expanded=True):
                u = st.text_input("Username", key="auth_uname", placeholder="e.g. pro_analyst")
                p = st.text_input("Password", type="password", key="auth_pword")
                if st.button("Sign In", type="primary", use_container_width=True):
                    if login(u, p, cfg):
                        st.success("Welcome back!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
        st.divider()


def render_login_dialog_button(cfg) -> None:
    """Render an inline pop-up login trigger right in the main page."""
    init_auth(cfg)
    if not st.session_state["authenticated"]:
        # If modern Streamlit supports st.dialog, use it; otherwise fallback to an expander
        if hasattr(st, "dialog"):
            @st.dialog("🔐 Sign In to PhishIntentionLLM Pro")
            def login_modal():
                st.write("Enter your credentials to unlock live screenshot analysis and VLM execution:")
                u = st.text_input("Username", key="modal_user", placeholder="pro_analyst")
                p = st.text_input("Password", type="password", key="modal_pass")
                if st.button("Authenticate", type="primary", use_container_width=True):
                    if login(u, p, cfg):
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

            if st.button("🔑 Sign In to Pro Plan", type="primary"):
                login_modal()
        else:
            with st.expander("🔑 Click here to Sign In with a Pro Account", expanded=False):
                c1, c2, c3 = st.columns([1.5, 1.5, 1])
                with c1:
                    u = st.text_input("Username", key="inline_user", placeholder="pro_analyst")
                with c2:
                    p = st.text_input("Password", type="password", key="inline_pass")
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Login", type="primary", use_container_width=True):
                        if login(u, p, cfg):
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")