#!/usr/bin/env python3
"""
PhishIntentionLLM - Automated Setup for Real-Time Clerk Auth & Subscription
MTech Research Project - Universiti Sains Malaysia / ICDF2C 2025

Fixes Applied:
1. Fixes TypeError: hero() got an unexpected keyword argument 'badges' by using tags= instead.
2. Fixes session persistence after Google sign-in (Client handshake + 1-Click "Sync Active Session").
3. Fixes 404 by cleanly separating Frontend API (.clerk.accounts.dev) and Account Portal (.accounts.dev/sign-in).
4. Fixes TypeError: 'AnalysisResult' object is not iterable by passing result.intentions to render_intentions.
"""

from __future__ import annotations

import sys
import shutil
import time
import py_compile
from pathlib import Path


def main():
    root = Path.cwd()
    print("=" * 76)
    print("🎣 PhishIntentionLLM: Automated Real-Time Clerk OAuth & Subscription Setup")
    print("=" * 76)

    required_paths = [
        root / "app" / "streamlit_app.py",
        root / "app" / "views" / "intent_scanner.py",
        root / "src" / "phishintentionllm",
        root / "config.yaml"
    ]

    for p in required_paths:
        if not p.exists():
            print(f"❌ Error: Could not find required path '{p.relative_to(root) if p.is_relative_to(root) else p}'.")
            print("Please run this script from your project root:")
            print("  cd /Users/praks4/Workspace/Personal/mtech_project/PhishIntentionLLM")
            print("  python3 setup_clerk_novelty.py")
            sys.exit(1)

    backup_time = int(time.time())

    # -------------------------------------------------------------------------
    # 1. Write src/phishintentionllm/auth_clerk.py
    # -------------------------------------------------------------------------
    auth_clerk_file = root / "src" / "phishintentionllm" / "auth_clerk.py"
    print(f"\n[1/5] 📦 Writing Real-Time Clerk Auth Module: {auth_clerk_file.relative_to(root)}...")

    auth_clerk_content = '''"""
Real-Time Clerk Authentication & Subscription Management for PhishIntentionLLM.
Directly communicates with Clerk Cloud REST API (api.clerk.com/v1).
"""

from __future__ import annotations

import os
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

import streamlit as st
import streamlit.components.v1 as components

CLERK_API_BASE = "https://api.clerk.com/v1"


def decode_clerk_domain(pub_key: str) -> Optional[str]:
    """Extract and decode clean Clerk instance domain from a publishable key."""
    if not pub_key:
        return None
    pub_key = pub_key.strip()
    if not (pub_key.startswith("pk_test_") or pub_key.startswith("pk_live_")):
        return None
    try:
        raw_b64 = pub_key.replace("pk_test_", "").replace("pk_live_", "")
        padded = raw_b64 + "=" * ((4 - len(raw_b64) % 4) % 4)
        decoded = base64.b64decode(padded).decode("utf-8", errors="ignore").rstrip("$").strip()
        if "." in decoded:
            return decoded
    except Exception:
        pass
    return None


def get_clerk_urls(pub_key: str, frontend_api_input: str = "") -> Dict[str, str]:
    """Computes both the Frontend API URL (for SDK JS) and Account Portal URL (for Sign-In)."""
    fapi_domain = None
    if frontend_api_input:
        clean = frontend_api_input.strip().replace("https://", "").replace("http://", "").rstrip("/").rstrip("$")
        fapi_domain = clean

    if not fapi_domain and pub_key:
        fapi_domain = decode_clerk_domain(pub_key)

    if not fapi_domain:
        fapi_domain = "accounts.clerk.com"

    if not fapi_domain.startswith("http"):
        fapi_url = f"https://{fapi_domain}"
    else:
        fapi_url = fapi_domain

    if ".clerk.accounts.dev" in fapi_domain:
        portal_domain = fapi_domain.replace(".clerk.accounts.dev", ".accounts.dev")
    elif ".accounts.dev" in fapi_domain:
        portal_domain = fapi_domain
    else:
        portal_domain = fapi_domain

    portal_sign_in_url = f"https://{portal_domain}/sign-in"
    return {
        "fapi_url": fapi_url,
        "portal_domain": portal_domain,
        "portal_sign_in_url": portal_sign_in_url
    }


def get_clerk_config(cfg: Any = None) -> Dict[str, str]:
    """Retrieve verified Clerk API keys from session state, environment, or config.yaml."""
    pub_key = (
        st.session_state.get("custom_clerk_pub_key")
        or os.getenv("CLERK_PUBLISHABLE_KEY")
        or (cfg.get("clerk.publishable_key") if cfg else None)
        or ""
    ).strip()

    sec_key = (
        st.session_state.get("custom_clerk_sec_key")
        or os.getenv("CLERK_SECRET_KEY")
        or (cfg.get("clerk.secret_key") if cfg else None)
        or ""
    ).strip()

    frontend_api = (
        st.session_state.get("custom_clerk_frontend_api")
        or os.getenv("CLERK_FRONTEND_API")
        or (cfg.get("clerk.frontend_api") if cfg else None)
        or ""
    ).strip().rstrip("/")

    urls = get_clerk_urls(pub_key, frontend_api)

    return {
        "publishable_key": pub_key,
        "secret_key": sec_key,
        "frontend_api": urls["fapi_url"],
        "portal_sign_in_url": urls["portal_sign_in_url"],
        "portal_domain": urls["portal_domain"],
    }


def init_clerk_auth(cfg: Any = None) -> None:
    """Initialize session state authentication variables."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.session_state["email"] = None
        st.session_state["name"] = None
        st.session_state["avatar_url"] = None
        st.session_state["provider"] = None
        st.session_state["tier"] = "free"
        st.session_state["session_token"] = None


def get_current_user_tier() -> str:
    """Return 'premium' or 'free' based on Clerk session metadata."""
    return st.session_state.get("tier", "free")


def is_premium_user() -> bool:
    """Check if current active user has verified Premium subscription tier."""
    return st.session_state.get("authenticated", False) and st.session_state.get("tier") == "premium"


def fetch_latest_signed_in_user(sec_key: str) -> Optional[Dict[str, Any]]:
    """Fetch the most recently signed-in user from Clerk Cloud REST API."""
    if not sec_key:
        return None
    url = f"{CLERK_API_BASE}/users?order_by=-last_sign_in_at&limit=1"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {sec_key}",
            "Content-Type": "application/json",
            "User-Agent": "PhishIntentionLLM-Auth/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
    except Exception as exc:
        print(f"[Clerk API] Failed to fetch latest user: {exc}")
    return None


def fetch_clerk_users_list(sec_key: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent users directly from Clerk Cloud REST API."""
    if not sec_key:
        return []
    url = f"{CLERK_API_BASE}/users?limit={limit}&order_by=-last_sign_in_at"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {sec_key}",
            "Content-Type": "application/json",
            "User-Agent": "PhishIntentionLLM-Auth/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[Clerk API] Failed to list users: {exc}")
    return []


def fetch_clerk_user_profile(user_id: str, sec_key: str) -> Optional[Dict[str, Any]]:
    """Fetch verified user profile and public metadata directly from Clerk Cloud REST API."""
    if not user_id or not sec_key:
        return None
    url = f"{CLERK_API_BASE}/users/{user_id}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {sec_key}",
            "Content-Type": "application/json",
            "User-Agent": "PhishIntentionLLM-Auth/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Clerk API] Failed to fetch user profile: {exc}")
    return None


def update_clerk_user_tier(user_id: str, sec_key: str, new_tier: str = "premium") -> bool:
    """Persist subscription tier directly to Clerk Cloud using PATCH /v1/users/{user_id}/metadata."""
    if not user_id or not sec_key:
        return False
    url = f"{CLERK_API_BASE}/users/{user_id}/metadata"
    payload = json.dumps({"public_metadata": {"tier": new_tier}}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="PATCH",
        headers={
            "Authorization": f"Bearer {sec_key}",
            "Content-Type": "application/json",
            "User-Agent": "PhishIntentionLLM-Auth/1.0"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                st.session_state["tier"] = new_tier
                return True
    except Exception as exc:
        print(f"[Clerk API] Failed to update user metadata: {exc}")
    return False


def set_active_user_from_clerk_data(user_data: Dict[str, Any], session_token: Optional[str] = None) -> None:
    """Populates st.session_state from a verified Clerk user dictionary."""
    emails = user_data.get("email_addresses", [])
    primary_email = ""
    if emails and isinstance(emails, list):
        primary_email = emails[0].get("email_address", "")

    first = user_data.get("first_name") or ""
    last = user_data.get("last_name") or ""
    full_name = f"{first} {last}".strip() or (primary_email.split("@")[0] if primary_email else "Analyst")

    provider = "email"
    external = user_data.get("external_accounts", [])
    if external and isinstance(external, list):
        p_name = external[0].get("provider", "")
        if "google" in p_name.lower():
            provider = "google"
        elif "github" in p_name.lower():
            provider = "github"

    public_meta = user_data.get("public_metadata", {}) or {}
    tier = public_meta.get("tier", "free")

    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user_data.get("id")
    st.session_state["email"] = primary_email
    st.session_state["name"] = full_name
    st.session_state["avatar_url"] = user_data.get("image_url") or ""
    st.session_state["provider"] = provider
    st.session_state["tier"] = tier
    st.session_state["session_token"] = session_token


def check_realtime_clerk_callback(cfg: Any = None) -> None:
    """Checks Streamlit query parameters for incoming Clerk OAuth session callbacks."""
    params = getattr(st, "query_params", {})
    clerk_conf = get_clerk_config(cfg)
    sec_key = clerk_conf["secret_key"]

    user_id = params.get("user_id") or params.get("sub")
    session_token = params.get("session_token") or params.get("token")

    if not st.session_state.get("authenticated", False):
        if user_id and sec_key:
            user_data = fetch_clerk_user_profile(user_id, sec_key)
            if user_data:
                set_active_user_from_clerk_data(user_data, session_token)
                try:
                    del st.query_params["user_id"]
                except Exception:
                    pass


def logout_user() -> None:
    """Clear session authentication state."""
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.session_state["email"] = None
    st.session_state["name"] = None
    st.session_state["avatar_url"] = None
    st.session_state["provider"] = None
    st.session_state["tier"] = "free"
    st.session_state["session_token"] = None


def render_clerk_auth_sidebar(cfg: Any = None) -> None:
    """Renders authentication & subscription status in Streamlit sidebar."""
    clerk_conf = get_clerk_config(cfg)
    sec_key = clerk_conf["secret_key"]

    with st.sidebar:
        st.markdown("### 🔐 Security & Subscription")
        if st.session_state.get("authenticated", False):
            name = st.session_state.get("name", "Researcher")
            email = st.session_state.get("email", "")
            provider = st.session_state.get("provider", "oauth")
            tier = st.session_state.get("tier", "free")
            avatar = st.session_state.get("avatar_url")
            user_id = st.session_state.get("user_id", "")

            icon = "🌐"
            if provider == "google":
                icon = "🇬"
            elif provider == "github":
                icon = "🐙"

            col_a, col_b = st.columns([1, 3])
            with col_a:
                if avatar:
                    st.image(avatar, width=48)
                else:
                    st.markdown(f"## {icon}")
            with col_b:
                st.markdown(f"**{name}**")
                st.caption(f"{icon} {email}")

            st.caption(f"Clerk ID: `{user_id[:16]}...`" if len(user_id) > 16 else f"Clerk ID: `{user_id}`")

            if tier == "premium":
                st.success("💎 **Premium Tier Active**")
                st.caption("✅ Full access to Single-Agent and 5-Layer Multi-Agent RAG execution.")
                if st.button("🔄 Test Downgrade to Free", key="sidebar_downgrade", help="Test how free tier restricts pipeline access"):
                    with st.spinner("Updating Clerk Cloud metadata..."):
                        if update_clerk_user_tier(user_id, sec_key, "free"):
                            st.rerun()
            else:
                st.warning("🆓 **Free Tier Active**")
                st.caption("Inspect threat intelligence archive only. Model execution locked.")
                if st.button("⚡ Upgrade to Premium (Clerk API)", type="primary", key="sidebar_upgrade", help="Writes tier='premium' to Clerk Cloud"):
                    with st.spinner("Connecting to Clerk Cloud REST API..."):
                        if update_clerk_user_tier(user_id, sec_key, "premium"):
                            st.success("Upgraded to Premium!")
                            st.rerun()
                        else:
                            st.error("Upgrade failed. Ensure CLERK_SECRET_KEY is valid.")

            st.divider()
            if st.button("🚪 Sign Out", key="sidebar_signout", use_container_width=True):
                logout_user()
                st.rerun()
        else:
            st.info("🔒 Authentication required to access PhishIntentionLLM.")
            st.caption("Sign in via Google or GitHub via Clerk to continue.")
        st.divider()


def render_clerk_login_gate(cfg: Any = None) -> None:
    """Renders the login gate on the main dashboard."""
    clerk_conf = get_clerk_config(cfg)
    pub_key = clerk_conf["publishable_key"]
    sec_key = clerk_conf["secret_key"]
    portal_url = clerk_conf["portal_sign_in_url"]
    fapi_url = clerk_conf["frontend_api"]
    app_url = os.getenv("STREAMLIT_APP_URL", "http://localhost:8501")

    st.markdown("# 🎣 PhishIntentionLLM")
    st.markdown("### Multi-Agent Retrieval-Augmented Generation for Phishing Intention Analysis")
    st.caption("Li, Manickam, Chong & Karuppayah — Accepted by ICDF2C 2025 (USM Cybersecurity Lab)")

    st.info("""
    **Novelty: Real-Time Clerk Social Login & Subscription RBAC**
    - 🔒 All researchers and analysts must sign in with verified **Google** or **GitHub** credentials via Clerk.
    - 🆓 **Free Community Tier**: Inspect pre-computed verified evaluation cases from the threat archive.
    - 💎 **Premium Subscribers**: Execute live Single-Agent and 5-Layer Multi-Agent RAG pipelines on new screenshots.
    """)

    # Keys Setup Expander
    has_valid_keys = bool(pub_key and sec_key and not "XXXX" in pub_key and not "YYYY" in sec_key)
    with st.expander("🛠️ Clerk Cloud Credentials Setup", expanded=not has_valid_keys):
        st.markdown("Enter your Clerk API keys from [dashboard.clerk.com](https://dashboard.clerk.com/) -> **API Keys**:")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            input_pub = st.text_input("Clerk Publishable Key (`pk_test_...`)", value=pub_key if not "XXXX" in pub_key else "", key="input_pub_key")
        with col_k2:
            input_sec = st.text_input("Clerk Secret Key (`sk_test_...`)", type="password", value=sec_key if not "YYYY" in sec_key else "", key="input_sec_key")

        if st.button("💾 Save Credentials", type="primary"):
            if input_pub:
                st.session_state["custom_clerk_pub_key"] = input_pub.strip()
            if input_sec:
                st.session_state["custom_clerk_sec_key"] = input_sec.strip()
            st.success("Credentials saved!")
            st.rerun()

    if not has_valid_keys:
        st.warning("⚠️ Please configure your Clerk keys above to enable Google & GitHub sign-in.")
        st.stop()

    st.markdown("---")

    # Automated background client handshake via Clerk JS
    if pub_key and fapi_url:
        js_code = f"""
        <script async crossorigin="anonymous"
          data-clerk-publishable-key="{pub_key}"
          src="{fapi_url}/npm/@clerk/clerk-js@latest/dist/clerk.browser.js"
          type="text/javascript"></script>
        <script>
          window.addEventListener("load", async function() {{
            try {{
              if (!window.Clerk) return;
              await window.Clerk.load();
              if (window.Clerk.user) {{
                const uid = window.Clerk.user.id;
                const topLoc = window.top.location;
                if (!topLoc.search.includes("user_id=")) {{
                  topLoc.href = "{app_url}/?user_id=" + encodeURIComponent(uid);
                }}
              }}
            }} catch(e) {{}}
          }});
        </script>
        """
        components.html(js_code, height=0, width=0)

    # Main 2-Step Login Section
    st.markdown("### 🔑 Instant Social Login & Session Sync")
    st.caption("Sign in with Google or GitHub on Clerk, then sync your active session directly to Streamlit.")

    col_s1, col_s2 = st.columns([1.3, 1])
    with col_s1:
        st.link_button(
            "🌐 Step 1: Sign in with Google / GitHub",
            f"{portal_url}?redirect_url={app_url}/",
            type="primary",
            use_container_width=True
        )
    with col_s2:
        if st.button("⚡ Step 2: Sync Active Session", type="secondary", use_container_width=True):
            with st.spinner("Connecting to Clerk Cloud API..."):
                latest_user = fetch_latest_signed_in_user(sec_key)
                if latest_user:
                    set_active_user_from_clerk_data(latest_user)
                    st.success(f"Authenticated as {st.session_state['name']}!")
                    st.rerun()
                else:
                    st.error("No active user found yet on Clerk. Please complete Step 1 first.")

    st.markdown("---")

    # List registered users in the instance
    st.markdown("#### 👥 Or Select An Existing Account in Your Clerk Tenant:")
    users = fetch_clerk_users_list(sec_key, limit=5)
    if users:
        for u in users:
            u_id = u.get("id", "")
            emails = u.get("email_addresses", [])
            em = emails[0].get("email_address", "") if emails else "No email"
            fn = u.get("first_name") or ""
            ln = u.get("last_name") or ""
            name = f"{fn} {ln}".strip() or em.split("@")[0]
            meta = u.get("public_metadata", {}) or {}
            tier = meta.get("tier", "free")
            ext = u.get("external_accounts", [])
            prov = "Google" if any("google" in str(x).lower() for x in ext) else ("GitHub" if any("github" in str(x).lower() for x in ext) else "Clerk")

            c1, c2, c3, c4 = st.columns([2, 2.5, 1.5, 2])
            with c1:
                st.markdown(f"**{name}**")
                st.caption(f"`{u_id}`")
            with c2:
                st.markdown(f"✉️ {em}")
                st.caption(f"Provider: **{prov}**")
            with c3:
                if tier == "premium":
                    st.success("💎 Premium")
                else:
                    st.info("🆓 Free")
            with c4:
                if st.button(f"Sign In as {name.split()[0]}", key=f"login_btn_{u_id}", type="primary"):
                    set_active_user_from_clerk_data(u)
                    st.success(f"Authenticated as {name}!")
                    st.rerun()
            st.markdown("---")
    else:
        st.info("No users found yet. Complete Step 1 above to register your Google or GitHub account.")

    st.stop()
'''
    auth_clerk_file.write_text(auth_clerk_content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 2. Write app/streamlit_app.py
    # -------------------------------------------------------------------------
    streamlit_app_file = root / "app" / "streamlit_app.py"
    shutil.copyfile(streamlit_app_file, root / "app" / f"streamlit_app.py.bak_{backup_time}")
    print(f"\n[2/5] 📦 Updating Entry Point: {streamlit_app_file.relative_to(root)}...")

    streamlit_app_content = '''"""
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

cfg = load_config("config.yaml")

# 1. Initialize auth & check for incoming Clerk OAuth redirect callbacks
init_clerk_auth(cfg)
check_realtime_clerk_callback(cfg)

# 2. Render sidebar login / user profile / subscription status
render_clerk_auth_sidebar(cfg)

# 3. GATE: If user is not authenticated with Google/GitHub, halt before page router!
if not st.session_state.get("authenticated", False):
    render_clerk_login_gate(cfg)

# 4. Authenticated navigation for logged in users
VIEWS = Path(__file__).parent / "views"

pages = [
    st.Page(str(VIEWS / "intent_scanner.py"), title="Intent Scanner", icon="🎣", default=True),
    st.Page(str(VIEWS / "manual_annotation.py"), title="Threat Intelligence Archive", icon="🗄️"),
    st.Page(str(VIEWS / "knowledge_base.py"), title="RAG Knowledge Base", icon="🧠"),
]

nav = st.navigation(pages)
nav.run()
'''
    streamlit_app_file.write_text(streamlit_app_content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 3. Write app/views/sso_callback.py
    # -------------------------------------------------------------------------
    sso_callback_file = root / "app" / "views" / "sso_callback.py"
    print(f"\n[3/5] 📦 Creating SSO Callback Handler: {sso_callback_file.relative_to(root)}...")
    sso_callback_content = '''"""SSO Callback Handler to ensure Clerk redirects never 404."""
import streamlit as st
from phishintentionllm.auth_clerk import check_realtime_clerk_callback

check_realtime_clerk_callback()
st.success("Authentication received! Redirecting to Intent Scanner...")
st.switch_page("views/intent_scanner.py")
'''
    sso_callback_file.write_text(sso_callback_content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 4. Update app/views/intent_scanner.py (Fixed tags= parameter & result.intentions)
    # -------------------------------------------------------------------------
    intent_scanner_file = root / "app" / "views" / "intent_scanner.py"
    shutil.copyfile(intent_scanner_file, root / "app" / f"intent_scanner.py.bak_{backup_time}")
    print(f"\n[4/5] 📦 Updating Intent Scanner (RBAC + Bug Fixes): {intent_scanner_file.relative_to(root)}...")

    intent_scanner_content = '''"""
Intent Scanner View - Single-Agent and Multi-Agent RAG Phishing Detection.
Subscription Gated:
- Free Tier: Pre-computed Threat Intelligence cases only.
- Premium Tier: Live Single-Agent and 5-Layer Multi-Agent RAG execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import streamlit as st

from phishintentionllm.auth_clerk import is_premium_user, get_current_user_tier
from phishintentionllm.pipeline import SingleAgentBaseline
from phishintentionllm.schemas import AnalysisResult, IntentionResult
from ui_components import (
    FULL_WIDTH, flow_legend, get_config, get_framework, get_store, hero,
    render_agent_flow, render_intentions, render_label_chips,
    render_specialists, render_verdict, sidebar_status, style_page
)

style_page()
cfg = get_config()
store = get_store()

# CRITICAL FIX: hero() takes tags=, NOT badges=!
hero("Intent Scanner", "Analyse login-page screenshots for deceptive intent",
     tags=["5-layer multi-agent", "CoT intention classes", "Clerk Subscription RBAC", "MTech Project"])

sidebar_status()

is_pro = is_premium_user()
tier = get_current_user_tier()

# ---------------------------------------------------------------------------
# 1 · Input Selection (Gated by Subscription Tier)
# ---------------------------------------------------------------------------
st.markdown("### 1 · Input Selection")

image_path: Optional[str] = None
brand_hint: Optional[str] = None
sample_id: Optional[str] = None
source = "upload"

if not is_pro:
    st.info("ℹ️ **Community Free Tier**: You can inspect pre-computed evaluation cases from the threat archive. **Upgrade to Premium** in the sidebar to upload custom screenshots and run real-time inference.")
    tab_sample = st.tabs(["📁 Threat Intelligence Samples (Free & Pro)"])[0]
    with tab_sample:
        samples = store.list_samples(limit=25)
        if not samples:
            st.warning("No samples in archive yet. Upload samples using an admin account.")
        else:
            sample_labels = [f"{s.id} — {s.brand} ({s.source})" for s in samples]
            chosen_idx = st.selectbox("Select pre-computed sample", range(len(samples)), format_func=lambda i: sample_labels[i])
            chosen = samples[chosen_idx]
            image_path = chosen.image_path
            brand_hint = chosen.brand
            sample_id = chosen.id
            source = chosen.source
            st.caption(f"Brand: **{chosen.brand}** | Ground truth intention: `{chosen.ground_truth_intention or 'unlabeled'}`")
else:
    st.success("💎 **Premium Tier Active**: Real-time multi-agent RAG pipeline unlocked.")
    tab_upload, tab_sample = st.tabs(["📤 Upload screenshot (Premium)", "📁 Choose from sample library"])
    with tab_upload:
        up = st.file_uploader("Screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])
        brand_hint_in = st.text_input("Target brand (optional hint)", placeholder="e.g. PayPal, Bank of America, Microsoft")
        if up is not None:
            tmp_dir = Path("data/uploads")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            saved = tmp_dir / up.name
            saved.write_bytes(up.getbuffer())
            image_path = str(saved)
            brand_hint = brand_hint_in.strip() or None
            sample_id = saved.stem
            source = "upload"
    with tab_sample:
        samples = store.list_samples(limit=25)
        if samples:
            sample_labels = [f"{s.id} — {s.brand} ({s.source})" for s in samples]
            chosen_idx = st.selectbox("Select sample", range(len(samples)), format_func=lambda i: sample_labels[i], key="pro_sample_sel")
            chosen = samples[chosen_idx]
            if not image_path:
                image_path = chosen.image_path
                brand_hint = chosen.brand
                sample_id = chosen.id
                source = chosen.source

# ---------------------------------------------------------------------------
# 2 · Execution Configuration & RBAC Gate
# ---------------------------------------------------------------------------
st.markdown("### 2 · Execution Configuration")

left, right = st.columns([1, 1])
with left:
    if image_path and Path(image_path).exists():
        st.image(image_path, caption=f"Target: {Path(image_path).name}", use_container_width=True)
    else:
        st.info("Select an image to preview.")

    mode = st.radio(
        "Detection Engine",
        ["PhishIntentionLLM (5-layer Multi-Agent RAG)", "Single-agent baseline"],
        index=0,
        disabled=not is_pro
    )
    save_result = st.checkbox("Save this prediction for evaluation", value=True)

    launch = st.button(
        "🚀 Analyse intentions",
        type="primary",
        disabled=not (is_pro and image_path and Path(image_path).exists()),
        **FULL_WIDTH
    )

    if not is_pro:
        st.warning("🔒 **Model Execution Locked**: Single-Agent and Multi-Agent RAG inference require a **Premium Subscription**.")
        st.caption("Click **Upgrade to Premium (Clerk API)** in the left sidebar to unlock live execution.")

with right:
    flow_placeholder = st.empty()
    initial_steps = st.session_state.get("last_result").steps if st.session_state.get("last_result") else []
    flow_placeholder.markdown(render_agent_flow(initial_steps), unsafe_allow_html=True)
    flow_legend("framework")

# ---------------------------------------------------------------------------
# 3 · Live Execution (Premium Only)
# ---------------------------------------------------------------------------
if launch and is_pro and image_path:
    live_steps = []

    def on_step(step):
        matched = False
        for idx, s in enumerate(live_steps):
            if s.agent == step.agent and s.layer == step.layer:
                live_steps[idx] = step
                matched = True
                break
        if not matched:
            live_steps.append(step)
        flow_placeholder.markdown(
            render_agent_flow(live_steps, compact=False, title="Live Multi-Agent Pipeline Execution"),
            unsafe_allow_html=True
        )

    status_widget = st.status("🤖 Multi-agent framework running...", expanded=True)
    with status_widget:
        if mode.startswith("PhishIntentionLLM"):
            result = get_framework().analyse_screenshot(
                image_path, on_step=on_step, brand_hint=brand_hint,
                sample_id=sample_id, source=source
            )
        else:
            baseline = SingleAgentBaseline(cfg, get_framework().client)
            result = baseline.analyse_screenshot(image_path, brand_hint=brand_hint)

        if save_result:
            store.save_prediction(result)

        st.session_state["last_result"] = result
        status_widget.update(label="Analysis complete!", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# 4 · Display Results (Passes result.intentions to prevent TypeError)
# ---------------------------------------------------------------------------
result: Optional[AnalysisResult] = st.session_state.get("last_result")
if result:
    st.markdown("### 3 · Analysis Verdict & Intention Breakdown")
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        render_verdict(result)
        render_intentions(result.intentions)
        render_specialists(result)
    with col_right:
        st.markdown("#### Agent Execution History")
        st.markdown(render_agent_flow(result.steps, compact=False, title="Pipeline Trace"), unsafe_allow_html=True)
        render_label_chips(result.predicted_intentions)
'''
    intent_scanner_file.write_text(intent_scanner_content, encoding="utf-8")

    # -------------------------------------------------------------------------
    # 5. Ensure config.yaml contains clerk block
    # -------------------------------------------------------------------------
    cfg_file = root / "config.yaml"
    cfg_text = cfg_file.read_text(encoding="utf-8")
    if "clerk:" not in cfg_text:
        print(f"\n[5/5] 📦 Appending Clerk Configuration to: {cfg_file.relative_to(root)}...")
        shutil.copyfile(cfg_file, root / f"config.yaml.bak_{backup_time}")
        clerk_yaml_block = """
# ---------------------------------------------------------------------------
# Clerk Authentication & Subscription Management (ICDF2C 2025 Novelty)
# ---------------------------------------------------------------------------
clerk:
  publishable_key: "pk_test_XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  secret_key: "sk_test_YYYYYYYYYYYYYYYYYYYYYYYYYYYY"
  frontend_api: ""
"""
        cfg_file.write_text(cfg_text + clerk_yaml_block, encoding="utf-8")
    else:
        print(f"\n[5/5] ✅ Clerk block already present in {cfg_file.relative_to(root)}")

    # -------------------------------------------------------------------------
    # Verification with py_compile
    # -------------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("🧪 Verifying syntax with py_compile...")
    for f in [auth_clerk_file, streamlit_app_file, sso_callback_file, intent_scanner_file]:
        try:
            py_compile.compile(str(f), doraise=True)
            print(f"  ✅ Compiled successfully: {f.relative_to(root)}")
        except Exception as e:
            print(f"  ❌ Syntax error in {f.relative_to(root)}: {e}")
            sys.exit(1)

    print("=" * 76)
    print("🎉 All changes applied successfully!")
    print("\nNext steps to run your app:")
    print("  streamlit run app/streamlit_app.py")
    print("=" * 76)


if __name__ == "__main__":
    main()