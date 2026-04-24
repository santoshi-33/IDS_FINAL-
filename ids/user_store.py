from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Dict, Tuple

USER_FILE_ENV = "IDS_USER_FILE"
DEFAULT_REL_PATH = "data/app_users.json"
PBKDF2_ITERS = 200_000


def _user_file_path(project_root: Path) -> Path:
    override = (os.getenv(USER_FILE_ENV) or "").strip()
    if override:
        return Path(override)
    return project_root / DEFAULT_REL_PATH


def _load_store(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {"users": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "users" not in data or not isinstance(data["users"], dict):
        return {"users": {}}
    return data


def _save_store(path: Path, store: Dict[str, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERS, dklen=32
    )
    return base64.b64encode(dk).decode("ascii")


def _verify_password(password: str, salt_b64: str, phash: str) -> bool:
    try:
        salt = base64.b64decode(salt_b64.encode("ascii"))
    except Exception:
        return False
    return secrets.compare_digest(_hash_password(password, salt), phash)


def sign_up(project_root: Path, email: str, password: str) -> Tuple[bool, str]:
    email = email.strip().lower()
    path = _user_file_path(project_root)
    store = _load_store(path)
    if email in store["users"]:
        return False, "This email is already registered. Use Login."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    salt = os.urandom(16)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    phash = _hash_password(password, salt)
    store["users"][email] = {"salt": salt_b64, "hash": phash, "algo": "pbkdf2_sha256"}
    _save_store(path, store)
    return True, "Account created. You can login now."


def verify_user(project_root: Path, email: str, password: str) -> Tuple[bool, str]:
    email = email.strip().lower()
    path = _user_file_path(project_root)
    store = _load_store(path)
    u = store["users"].get(email)
    if not u:
        return False, "Unknown email. Please sign up first."
    if _verify_password(password, u["salt"], u["hash"]):
        return True, "ok"
    return False, "Wrong password."


def env_bootstrap_exists() -> bool:
    """If IDS_USER+IDS_PASS are set, allow legacy one-account login (optional)."""
    u = (os.getenv("IDS_USER") or "").strip()
    p = (os.getenv("IDS_PASS") or "").strip()
    return bool(u and p)
