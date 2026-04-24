"""
Signed browser session cookie for Streamlit: stay logged in across refresh / new visits.

Set ``IDS_SESSION_SECRET`` in ``.streamlit/secrets.toml`` (or env) in production; otherwise a dev default is used.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Optional

# Cookie name (must match JS in streamlit_app)
COOKIE_NAME = "ids_session"
DEFAULT_TTL_SEC = 30 * 24 * 60 * 60  # 30 days


def get_signing_key() -> bytes:
    """HMAC key derived from config (lazy import of Streamlit ok in apps)."""
    s = ""
    try:
        import streamlit as st  # type: ignore

        if hasattr(st, "secrets") and st.secrets:
            try:
                s = str(st.secrets["IDS_SESSION_SECRET"]).strip()  # type: ignore[index]
            except Exception:
                try:
                    s = (st.secrets.get("IDS_SESSION_SECRET") or "").strip()  # type: ignore[union-attr]
                except Exception:
                    s = ""
    except Exception:
        s = ""
    if not s:
        s = (os.environ.get("IDS_SESSION_SECRET") or "").strip()
    if not s:
        s = "ids_dev_only_set_IDS_SESSION_SECRET"
    return hashlib.sha256(s.encode("utf-8")).digest()


def _normalize_email(s: str) -> str:
    t = (s or "").strip().lower()
    return t


def make_session_token(email: str, *, ttl_sec: int = DEFAULT_TTL_SEC) -> str:
    email = _normalize_email(email)
    exp = int(time.time()) + int(ttl_sec)
    key = get_signing_key()
    body = f"{email}\n{exp}"
    mac = hmac.new(key, body.encode("utf-8"), hashlib.sha256).hexdigest()
    token = f"{body}\n{mac}"
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii")


def parse_session_token(token: str) -> Optional[str]:
    """Return email if token is valid and not expired, else None."""
    if not token or not token.strip():
        return None
    t = token.strip()
    # pad for urlsafe b64
    pad = (-len(t)) % 4
    t += "=" * pad
    try:
        raw = base64.urlsafe_b64decode(t)
        parts = raw.decode("utf-8").split("\n")
        if len(parts) != 3:
            return None
        email, exp_s, mac = parts[0], parts[1], parts[2]
        exp = int(exp_s)
    except Exception:
        return None
    if int(time.time()) > exp:
        return None
    key = get_signing_key()
    body = f"{email}\n{exp}"
    expected = hmac.new(key, body.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, mac):
        return None
    return _normalize_email(email) if email else None
