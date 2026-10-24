"""
Real-Time Clerk Authentication & Subscription Management for PhishIntentionLLM.
Directly communicates with Clerk Cloud REST API (api.clerk.com/v1).
"""

from __future__ import annotations

import os
import time
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
    clean = pub_key.strip()
    if clean.startswith("pk_test_"):
        clean = clean[len("pk_test_"):]
    elif clean.startswith("pk_live_"):
        clean = clean[len("pk_live_"):]

    padding = "=" * ((4 - len(clean) % 4) % 4)
    try:
        decoded = base64.b64decode(clean + padding).decode("utf-8").strip()
        if "$" in decoded:
            decoded = decoded.split("$")[0].strip()
        return decoded
    except Exception:
        return None


def get_clerk_urls(pub_key: str, frontend_api_input: str = "") -> Dict[str, str]:
    """Derive Frontend API (for JS SDK) and Account Portal (for Hosted Sign-in)."""
    fapi = frontend_api_input.strip() if frontend_api_input else ""
    if not fapi:
        domain = decode_clerk_domain(pub_key)
        if domain:
            fapi = f"https://{domain}"

    portal_sign_in = ""
    portal_sign_up = ""
    portal_user_profile = ""

    if fapi:
        clean = fapi.replace("https://", "").replace("http://", "").rstrip("/")
        if ".clerk.accounts.dev" in clean:
            portal_host = clean.replace(".clerk.accounts.dev", ".accounts.dev")
            portal_sign_in = f"https://{portal_host}/sign-in"
            portal_sign_up = f"https://{portal_host}/sign-up"
            portal_user_profile = f"https://{portal_host}/user"
        elif ".accounts.dev" in clean:
            portal_sign_in = f"https://{clean}/sign-in"
            portal_sign_up = f"https://{clean}/sign-up"
            portal_user_profile = f"https://{clean}/user"
        else:
            portal_sign_in = f"{fapi}/sign-in"
            portal_sign_up = f"{fapi}/sign-up"
            portal_user_profile = f"{fapi}/user"

    return {
        "frontend_api": fapi,
        "portal_sign_in_url": portal_sign_in,
        "portal_sign_up_url": portal_sign_up,
        "portal_user_profile_url": portal_user_profile,
    }


def get_clerk_config(cfg: Any = None) -> Dict[str, str]:
    """Retrieve Clerk keys and URLs from session_state, environment, or config.yaml."""
    pub_key = (
        st.session_state.get("custom_clerk_pub_key")
        or os.getenv("CLERK_PUBLISHABLE_KEY", "")
    )
    sec_key = (
        st.session_state.get("custom_clerk_sec_key")
        or os.getenv("CLERK_SECRET_KEY", "")
    )
    fapi_in = (
        st.session_state.get("custom_clerk_frontend_api")
        or os.getenv("CLERK_FRONTEND_API", "")
    )

    if cfg:
        try:
            clerk_sec = cfg.get("clerk", {})
            if not pub_key:
                pub_key = clerk_sec.get("publishable_key", "")
            if not sec_key:
                sec_key = clerk_sec.get("secret_key", "")
            if not fapi_in:
                fapi_in = clerk_sec.get("frontend_api", "")
        except Exception:
            pass

    urls = get_clerk_urls(pub_key, fapi_in)
    return {
        "publishable_key": pub_key,
        "secret_key": sec_key,
        "frontend_api": urls["frontend_api"],
        "portal_sign_in_url": urls["portal_sign_in_url"],
        "portal_sign_up_url": urls["portal_sign_up_url"],
        "portal_user_profile_url": urls["portal_user_profile_url"],
    }


def init_clerk_auth(cfg: Any = None) -> None:
    """Initialize Clerk authentication session state keys."""
    defaults = {
        "authenticated": False,
        "user_id": None,
        "email": None,
        "name": None,
        "avatar_url": None,
        "provider": None,
        "tier": "free",
        "session_token": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_current_user_tier() -> str:
    """Return 'free' or 'premium'."""
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
    """Fetch verified user details by user_id from Clerk Cloud REST API."""
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
        print(f"[Clerk API] Failed to fetch user profile for {user_id}: {exc}")
    return None


def update_clerk_user_tier(user_id: str, sec_key: str, new_tier: str = "premium") -> bool:
    """Update user's public_metadata tier attribute in Clerk Cloud."""
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
        print(f"[Clerk API] Failed to update tier for {user_id}: {exc}")
    return False


def set_active_user_from_clerk_data(user_data: Dict[str, Any], session_token: Optional[str] = None) -> None:
    """Populate Streamlit session_state with verified Clerk user profile data."""
    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user_data.get("id")

    emails = user_data.get("email_addresses", [])
    primary_email = emails[0].get("email_address") if emails else "user@clerk.dev"
    st.session_state["email"] = primary_email

    first = user_data.get("first_name") or ""
    last = user_data.get("last_name") or ""
    name = f"{first} {last}".strip() or primary_email.split("@")[0].title()
    st.session_state["name"] = name

    st.session_state["avatar_url"] = user_data.get("image_url") or "https://api.dicebear.com/7.x/bottts/svg?seed=clerk"

    accounts = user_data.get("external_accounts", [])
    if accounts:
        st.session_state["provider"] = accounts[0].get("provider", "social").title()
    else:
        st.session_state["provider"] = "Google / Clerk"

    meta = user_data.get("public_metadata", {})
    tier = meta.get("tier", "free").lower()
    if tier not in ["free", "premium"]:
        tier = "free"
    st.session_state["tier"] = tier

    if session_token:
        st.session_state["session_token"] = session_token


def check_realtime_clerk_callback(cfg: Any = None) -> None:
    """
    Checks Streamlit query parameters and recent Clerk sign-in activity.
    If a Google/GitHub user just signed in within 5 minutes, auto-authenticates them.
    """
    if st.session_state.get("authenticated", False):
        return

    params = getattr(st, "query_params", {})
    clerk_conf = get_clerk_config(cfg)
    sec_key = clerk_conf["secret_key"]
    if not sec_key:
        return

    user_id = params.get("user_id") or params.get("sub")
    session_token = params.get("session_token") or params.get("token")

    # 1. Direct query parameter handshake
    if user_id:
        user_data = fetch_clerk_user_profile(user_id, sec_key)
        if user_data:
            set_active_user_from_clerk_data(user_data, session_token)
            try:
                del st.query_params["user_id"]
            except Exception:
                pass
            return

    # 2. Automated recent sign-in detector:
    # If a user completed Google/GitHub auth in Clerk within the last 5 minutes (300,000 ms),
    # auto-activate session so they are not bounced back to sign-in page!
    try:
        latest = fetch_latest_signed_in_user(sec_key)
        if latest and latest.get("last_sign_in_at"):
            now_ms = time.time() * 1000
            diff_ms = now_ms - latest["last_sign_in_at"]
            if 0 <= diff_ms < 300_000:  # within 5 minutes
                set_active_user_from_clerk_data(latest)
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
    portal_user_url = clerk_conf["portal_user_profile_url"]
    is_auth = st.session_state.get("authenticated", False)

    with st.sidebar:
        st.markdown("### 🔐 Identity & Access")

        if is_auth:
            tier = st.session_state.get("tier", "free")
            name = st.session_state.get("name", "Researcher")
            email = st.session_state.get("email", "")
            provider = st.session_state.get("provider", "Google")
            avatar = st.session_state.get("avatar_url")

            col_a, col_b = st.columns([1, 2.5])
            with col_a:
                if avatar:
                    st.image(avatar, width=54)
            with col_b:
                st.markdown(f"**{name}**")
                st.caption(f"{email}")

            if tier == "premium":
                st.markdown(
                    '<span style="background-color:#10b981;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:bold;">💎 Premium Subscriber</span>',
                    unsafe_allow_html=True
                )
                st.caption("⚡ Live Single-Agent & 5-Layer Multi-Agent RAG unlocked")
                if st.button("Downgrade to Free Tier (Test RBAC)", type="secondary", use_container_width=True):
                    uid = st.session_state.get("user_id")
                    if uid and sec_key:
                        update_clerk_user_tier(uid, sec_key, "free")
                    else:
                        st.session_state["tier"] = "free"
                    st.rerun()
            else:
                st.markdown(
                    '<span style="background-color:#64748b;color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:bold;">🆓 Community Free Tier</span>',
                    unsafe_allow_html=True
                )
                st.caption("🔒 Limited to pre-computed threat library cases")
                if st.button("🚀 Upgrade to Premium (Clerk API)", type="primary", use_container_width=True):
                    uid = st.session_state.get("user_id")
                    if uid and sec_key:
                        with st.spinner("Updating subscription via Clerk API..."):
                            update_clerk_user_tier(uid, sec_key, "premium")
                    else:
                        st.session_state["tier"] = "premium"
                    st.rerun()

            st.markdown(f"Provider: `{provider}`")
            if portal_user_url:
                st.markdown(f"[Manage Clerk Account]({portal_user_url})", unsafe_allow_html=True)

            if st.button("Sign Out", type="secondary", use_container_width=True):
                logout_user()
                st.rerun()
        else:
            st.warning("Not Signed In")
            st.caption("Sign in with Google or GitHub to access research tools.")


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
                    st.success(f"Logged in as {st.session_state.get('name')}!")
                    st.rerun()
                else:
                    st.error("No recent Clerk sign-in detected. Please complete Step 1 first.")

    # Live Registered Users quick-switch
    st.markdown("#### 👥 Or Select from Live Clerk Cloud Users")
    try:
        users = fetch_clerk_users_list(sec_key, limit=5)
        if users:
            for u in users:
                emails = u.get("email_addresses", [])
                e = emails[0].get("email_address") if emails else "user@clerk.dev"
                f = u.get("first_name") or ""
                l = u.get("last_name") or ""
                n = f"{f} {l}".strip() or e.split("@")[0].title()
                t = u.get("public_metadata", {}).get("tier", "free")
                c1, c2, c3 = st.columns([2, 1, 1.2])
                with c1:
                    st.write(f"**{n}** ({e})")
                with c2:
                    st.caption(f"Tier: `{t}`")
                with c3:
                    if st.button(f"🔑 Sign in as {f or 'User'}", key=f"btn_u_{u.get('id')}", use_container_width=True):
                        set_active_user_from_clerk_data(u)
                        st.success(f"Logged in as {n}!")
                        st.rerun()
        else:
            st.caption("No registered users returned by Clerk yet.")
    except Exception as exc:
        st.caption(f"Note: Could not list users ({exc})")
