from __future__ import annotations

import gzip
import ipaddress
import json
import os
import re
import shutil
import socket
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as st_components
from sklearn.metrics import confusion_matrix
import subprocess

# Ensure project root is importable when running via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ids.live import scapy_sniff_available, simulate_stream
from ids.pipeline import predict_df, predict_from_path_csv_in_chunks, predict_from_uploaded_csv_in_chunks
from ids.synthetic_bench import dataframe_nsl_synthetic, write_until_size
from ids.auth_session import COOKIE_NAME, make_session_token, parse_session_token
from ids.reporting import build_ids_report_pdf, load_metrics_json
from ids.user_store import env_bootstrap_exists, sign_up, verify_user


APP_TITLE = "ML-based Intrusion Detection System (IDS)"

# Benchmark CSVs: scripts/generate_test_datasets.py (--benchmark-tiers, --out-file, …)
TEST_CASES_DIR = PROJECT_ROOT / "data" / "test_cases"
# Must match .streamlit/config.toml [server] maxUploadSize (MB) for local/self-host
MAX_UPLOAD_MB = 2048
# Entire-file scan above this size uses chunked read + predict (supports ~1 GB uploads)
LARGE_UPLOAD_BYTES = 50 * 1024 * 1024
# "Test case files" page: do not offer st.file_uploader for huge tiers (browser upload = long buffer).
TEST_CASE_IN_PAGE_UPLOAD_MAX = 50 * 1024 * 1024
# In-browser download cap (results CSV); if larger we gzip or only offer sample
MAX_DOWNLOAD_RESULT_BYTES = 100 * 1024 * 1024
# Streamlit Community Cloud may cap lower — large files: generate on server, don’t upload
BENCHMARK_FILES: list[tuple[str, str, int]] = [
    ("~10 KB", "test_10kb.csv", 10 * 1024),
    ("~100 KB", "test_100kb.csv", 100 * 1024),
    ("~500 KB", "test_500kb.csv", 500 * 1024),
    ("~1 MB", "test_1mb.csv", 1 * 1024 * 1024),
    ("~5 MB", "test_5mb.csv", 5 * 1024 * 1024),
    ("~20 MB", "test_20mb.csv", 20 * 1024 * 1024),
    ("~100 MB", "test_100mb.csv", 100 * 1024 * 1024),
    ("~200 MB", "test_200mb.csv", 200 * 1024 * 1024),
    ("~1 GB (max tier)", "test_1gb.csv", 1024**3),
]


def _fmt_file_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def _save_upload_to_test_cases(uploaded, safe_name: str) -> None:
    """Write an uploaded file into TEST_CASES_DIR (safe basename only)."""
    name = Path(safe_name).name
    if name != safe_name or ".." in safe_name or Path(safe_name).is_absolute():
        raise ValueError("Invalid file name.")
    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)
    dest = TEST_CASES_DIR / name
    dest.write_bytes(uploaded.getvalue())


def _generate_test_case_file_inprocess(dest: Path, target_bytes: int) -> tuple[bool, str]:
    """Write a benchmark-size CSV in the Streamlit process (avoids failing subprocess/CLI on Cloud)."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_until_size(dest, int(target_bytes), np.random.default_rng(42))
        return True, ""
    except Exception as e:
        return False, str(e)


def _get_env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


# Practical pattern for campus emails (e.g. user@dept.skit.ac.in); avoids ghost "invalid" on Cloud
_EMAIL_RE = re.compile(r"^[\w.!#$%&'*+/=?^`{|}~-]+@(?:[\w-]+\.)+[\w-]{2,}$")


def _normalize_email(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\ufeff", "")
    t = t.replace("＠", "@")
    t = t.replace("﹫", "@")
    if "\r" in t or "\n" in t:
        t = t.splitlines()[0].strip()
    return t


def _is_valid_email(s: str) -> bool:
    t = _normalize_email(s)
    if not t or t.count("@") != 1:
        return False
    return bool(_EMAIL_RE.fullmatch(t))


def _read_ids_session_cookie() -> Optional[str]:
    """Read signed session cookie (Streamlit 1.38+ exposes ``st.context.cookies``)."""
    try:
        ctx = getattr(st, "context", None)
        if ctx is None:
            return None
        raw = getattr(ctx, "cookies", None)
        if not raw:
            return None
        v = raw.get(COOKIE_NAME) if isinstance(raw, dict) else None
        if v is None and not isinstance(raw, dict):
            try:
                v = raw[COOKIE_NAME]  # type: ignore[index]
            except Exception:
                v = None
        if v is None or not str(v).strip():
            return None
        return str(v).strip()
    except Exception:
        return None


def _emit_session_cookie_javascript(token: str) -> None:
    """Set browser cookie via hidden iframe; survives refresh. Server reads with ``st.context.cookies`` (Streamlit 1.38+)."""
    t_json = json.dumps(token)
    st_components.html(
        f"""<script>
(function(){{
  const t = {t_json};
  const name = {json.dumps(COOKIE_NAME)};
  const maxAge = 60 * 60 * 24 * 30;
  let c = name + '=' + encodeURIComponent(t) + '; max-age=' + maxAge + '; path=/; SameSite=Lax';
  if (window.location && window.location.protocol === 'https:') {{ c += '; Secure'; }}
  document.cookie = c;
}})();
</script>""",
        height=0,
        width=0,
    )


def _emit_clear_session_cookie_javascript() -> None:
    st_components.html(
        f"""<script>
(function(){{
  const name = {json.dumps(COOKIE_NAME)};
  let c = name + '=; max-age=0; path=/; SameSite=Lax';
  if (window.location && window.location.protocol === 'https:') {{ c += '; Secure'; }}
  document.cookie = c;
}})();
</script>""",
        height=0,
        width=0,
    )


def _try_restore_auth_from_cookie() -> bool:
    """If server sees a valid cookie, restore authed + email. Returns True if state changed."""
    if st.session_state.get("authed"):
        return False
    if st.session_state.get("ids_logout_clears_cookie"):
        return False
    cval = _read_ids_session_cookie()
    if not cval:
        return False
    email = parse_session_token(cval)
    if not email:
        return False
    st.session_state.authed = True
    st.session_state.user_email = email
    return True


def _persist_new_login_email(email: str) -> None:
    """Write cookie + mark session as logged in (call before ``st.rerun``)."""
    st.session_state.pop("ids_logout_clears_cookie", None)
    st.session_state.authed = True
    st.session_state.user_email = _normalize_email(email)
    _emit_session_cookie_javascript(make_session_token(st.session_state.user_email))


def check_login() -> bool:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

    if st.session_state.authed:
        return True

    if _try_restore_auth_from_cookie():
        return True

    st.title(APP_TITLE)
    st.subheader("Authentication")

    tab_signup, tab_login = st.tabs(["Sign up", "Login"])

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            su_email = st.text_input("Email", key="su_email", autocomplete="email", placeholder="you@university.edu")
            su_pw = st.text_input("Password", type="password", key="su_pw", autocomplete="new-password")
            su_pw2 = st.text_input(
                "Confirm password", type="password", key="su_pw2", autocomplete="new-password"
            )
            if st.form_submit_button("Create account", type="primary"):
                su_email = _normalize_email(su_email)
                if not _is_valid_email(su_email):
                    st.error("Please enter a valid email address (check for extra spaces or special @ character).")
                elif su_pw != su_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = sign_up(PROJECT_ROOT, su_email, su_pw)
                    if ok:
                        st.success(msg)
                        _persist_new_login_email(su_email)
                        st.rerun()
                    else:
                        st.error(msg)

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            lg_email = st.text_input("Email", key="lg_email", autocomplete="email", placeholder="you@university.edu")
            lg_pw = st.text_input("Password", type="password", key="lg_pw", autocomplete="current-password")
            if st.form_submit_button("Login", type="primary"):
                lg_email = _normalize_email(lg_email)
                if not _is_valid_email(lg_email):
                    st.error("Please enter a valid email address (check for extra spaces or special @ character).")
                else:
                    email_norm = lg_email.lower()

                    legacy_ok = False
                    if env_bootstrap_exists():
                        exp = _get_env("IDS_USER", "").strip().lower()
                        if email_norm == exp and lg_pw == _get_env("IDS_PASS", ""):
                            legacy_ok = True

                    ok, msg = verify_user(PROJECT_ROOT, lg_email, lg_pw)
                    if legacy_ok or ok:
                        _persist_new_login_email(lg_email)
                        st.rerun()
                    else:
                        st.error(msg)

    st.caption(
        "Passwords: `data/app_users.json` (hashed). **Stay signed in** uses a signed cookie (30 days); "
        "set **`IDS_SESSION_SECRET`** in [Streamlit secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started#secrets) for production. "
        "On Community Cloud, `app_users.json` can reset on redeploy—sign up again or use `IDS_USER` / `IDS_PASS`."
    )
    return False


def load_model(model_path: str) -> Any:
    bundle = joblib.load(model_path)
    return bundle["pipeline"]


def sidebar_model_picker() -> str:
    st.sidebar.header("Model")
    default = "models/ids_model.joblib"
    model_path = st.sidebar.text_input("Model path", value=default)
    if not Path(model_path).exists():
        st.sidebar.warning("Model file not found. Train first using `python -m ids.train ...`.")
    return model_path


def maybe_bootstrap_demo_assets(model_path: str) -> None:
    """
    Streamlit Community Cloud doesn't persist local artifacts unless committed.
    If the model is missing, offer a one-click bootstrap that downloads NSL-KDD,
    prepares `data/train.csv` + `data/test.csv`, and trains the model.
    """
    if Path(model_path).exists():
        return

    st.error("Model not found.")
    st.info(
        "To make the app work end-to-end on Streamlit Cloud, we can bootstrap the demo "
        "dataset and train the model automatically."
    )
    if st.button("Setup demo (download dataset + train model)"):
        with st.status("Setting up demo assets...", expanded=True) as status:
            try:
                cmd = [sys.executable, "scripts/setup_demo.py"]
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                st.code(proc.stdout or "Done.")
                status.update(label="Demo setup completed.", state="complete")
                st.success("Model trained. Reloading...")
                st.rerun()
            except subprocess.CalledProcessError as e:
                status.update(label="Demo setup failed.", state="error")
                st.code((e.stdout or "") + "\n" + (e.stderr or ""))
                st.error("Setup failed. Check logs above.")


def render_metrics() -> None:
    st.header("Dashboard")
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        c4.metric("F1", f"{metrics.get('f1', 0):.4f}")

        if metrics.get("roc_auc") is not None:
            st.caption(f"ROC-AUC: {metrics.get('roc_auc'):.6f}")

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and len(cm) == 2:
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(cm, index=["Actual Normal", "Actual Attack"], columns=["Pred Normal", "Pred Attack"])
            fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        rep = metrics.get("classification_report")
        if rep:
            with st.expander("Classification report"):
                st.code(rep)
    else:
        st.info("No metrics found at `models/metrics.json` yet.")

    st.divider()
    st.subheader("Report (PDF)")
    st.caption(
        "Build a PDF with training metrics (and the latest scan summary if you ran "
        "“Run Detection” in Upload & Scan in this session)."
    )
    c1, c2 = st.columns(2)
    with c1:
        gen = st.button("Generate PDF report", type="primary")
    with c2:
        st.caption("Tip: run a scan on Upload & Scan first for a richer report.")

    def _build_pdf_bytes() -> bytes:
        metrics = load_metrics_json("models/metrics.json")
        scan = st.session_state.get("last_scan_summary")
        return build_ids_report_pdf(
            title="Intrusion Detection System Report",
            user_email=str(st.session_state.get("user_email", "")),
            metrics=metrics,
            scan_summary=scan,
        )

    if gen:
        with st.spinner("Building PDF..."):
            st.session_state["last_report_pdf"] = _build_pdf_bytes()

    pdf_bytes: Optional[bytes] = st.session_state.get("last_report_pdf")  # type: ignore[assignment]
    if pdf_bytes:
        st.download_button(
            "Download last generated PDF",
            data=pdf_bytes,
            file_name="ids_report.pdf",
            mime="application/pdf",
        )


def _load_default_demo_df() -> Optional[pd.DataFrame]:
    demo_path = Path("data/test.csv")
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return None


def _read_streamlit_uploaded_csv(up: Any, *, nrows: Optional[int] = None) -> pd.DataFrame:
    """CSV or gzip-compressed CSV (`.csv.gz` / `.gz`) — smaller uploads for the same data."""
    name = (getattr(up, "name", "") or "").lower()
    if name.endswith(".gz"):
        return pd.read_csv(up, compression="gzip", nrows=nrows, low_memory=False)
    return pd.read_csv(up, nrows=nrows, low_memory=False)


def _uploaded_file_size_bytes(up: Any) -> int:
    s = getattr(up, "size", None)
    if isinstance(s, (int, float)) and s > 0:
        return int(s)
    try:
        up.seek(0, 2)
        n = int(up.tell())
        up.seek(0)
        return n
    except Exception:
        return 0


def _preview_upload_nrows(up: Any, n: int = 100) -> pd.DataFrame:
    up.seek(0)
    name = (getattr(up, "name", "") or "").lower()
    if name.endswith(".gz"):
        return pd.read_csv(up, compression="gzip", nrows=n, low_memory=False)
    return pd.read_csv(up, nrows=n, low_memory=False)


def _list_test_case_csvs() -> list[Path]:
    if not TEST_CASES_DIR.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(TEST_CASES_DIR.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith(".csv") or n.endswith(".gz"):
            out.append(p)
    return out


def _read_path_csv(path: Path, *, nrows: Optional[int] = None) -> pd.DataFrame:
    name = path.name.lower()
    kw: dict = {"low_memory": False}
    if name.endswith(".gz"):
        kw["compression"] = "gzip"
    if nrows is not None:
        kw["nrows"] = nrows
    return pd.read_csv(path, **kw)


def _clear_url_temp_in_session() -> None:
    p = st.session_state.pop("ids_url_temp_path", None)
    if isinstance(p, str) and os.path.isfile(p):
        try:
            os.remove(p)
        except OSError:
            pass


def _url_host_is_public(url: str) -> tuple[bool, str]:
    """Block obvious SSRF targets (localhost / private IPs)."""
    p = urlparse(url.strip())
    host = p.hostname
    if not host:
        return False, "Invalid URL (no host)."
    try:
        for res in socket.getaddrinfo(host, None):
            ipstr = res[4][0]
            ip = ipaddress.ip_address(ipstr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, "That host resolves to a private or local address (blocked)."
    except OSError as e:
        return False, f"Could not resolve host: {e}"
    return True, ""


def _download_url_to_tempfile(url: str) -> tuple[Optional[str], Optional[str]]:
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        return None, "Only http:// and https:// URLs are allowed."
    ok, msg = _url_host_is_public(url)
    if not ok:
        return None, msg
    path_part = (p.path or "").lower()
    ext = ".gz" if path_part.endswith((".gz", ".gzip")) else ".csv"
    max_b = MAX_UPLOAD_MB * 1024 * 1024
    fd, path = tempfile.mkstemp(prefix="ids_url_", suffix=ext)
    try:
        with os.fdopen(fd, "wb") as outf:
            req = Request(url.strip(), headers={"User-Agent": "IDS-Streamlit/1.0", "Accept": "*/*"}, method="GET")
            with urlopen(req, timeout=300) as resp:
                total = 0
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_b:
                        return None, f"Download exceeds {MAX_UPLOAD_MB} MB (app limit)."
                    outf.write(chunk)
    except Exception as e:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
        return None, str(e)
    return path, None


def _cleanup_pred_temp_keys() -> None:
    for _k in ("ids_pred_temp_csv", "ids_pred_gz_path"):
        _p = st.session_state.get(_k)
        if _p and isinstance(_p, str) and os.path.isfile(_p):
            try:
                os.remove(_p)
            except OSError:
                pass
        st.session_state.pop(_k, None)


def _write_scan_result_ui(out: pd.DataFrame, *, use_chunked: bool) -> None:
    _ = use_chunked
    st.subheader("Results (first rows)")
    st.dataframe(out.head(200), use_container_width=True)

    counts = out["prediction"].value_counts().reset_index()
    counts.columns = ["prediction", "count"]
    fig = px.bar(counts, x="prediction", y="count", title="Predicted Traffic Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary")
    total = int(len(out))
    attacks = int((out["prediction"] == "attack").sum())
    normals = total - attacks
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{total:,}")
    c2.metric("Normal", f"{normals:,}")
    c3.metric("Attack", f"{attacks:,}")

    if "label" in out.columns:
        try:
            y_true = out["label"].astype(str).str.lower().map({"normal": 0, "attack": 1})
            y_pred = out["prediction"].astype(str).str.lower().map({"normal": 0, "attack": 1})
            if y_true.notna().all() and y_pred.notna().all():
                cm_local = confusion_matrix(y_true, y_pred)
                st.session_state["last_scan_confusion"] = cm_local.tolist()
                cm_df = pd.DataFrame(
                    cm_local,
                    index=["Actual Normal", "Actual Attack"],
                    columns=["Pred Normal", "Pred Attack"],
                )
                fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix (if labels present)")
                st.plotly_chart(fig_cm, use_container_width=True)
        except Exception:
            pass

    if "attack_probability" in out.columns:
        fig2 = px.histogram(
            out,
            x="attack_probability",
            nbins=40,
            title="Attack probability distribution",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.download_button(
        "Download results CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="ids_results.csv",
        mime="text/csv",
    )

    st.session_state["last_scan_summary"] = {
        "rows": int(len(out)),
        "normal": int((out["prediction"] == "normal").sum()),
        "attack": int((out["prediction"] == "attack").sum()),
        "head_rows": out.head(15).to_dict(orient="records"),
        "has_labels": bool("label" in out.columns),
    }


def _write_scan_result_chunked_ui(
    summary: dict, prev: pd.DataFrame, tpath: str, lcol: Optional[str]
) -> None:
    _ = lcol
    st.subheader("Results (preview sample of rows + full scan counts)")
    st.caption(
        "**Every row in your file was scanned.** The table and charts below use the **first ~1,200** result rows for display only."
    )
    total = int(summary.get("rows", 0))
    if total <= 0:
        st.error("No rows were scanned.")
        return
    if prev is not None and not prev.empty:
        st.dataframe(prev.head(200), use_container_width=True)
    else:
        st.warning("No preview table (inner chunks empty); see totals below.")

    attacks = int(summary.get("attack", 0))
    normals = int(summary.get("normal", 0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows (full file)", f"{total:,}")
    c2.metric("Normal", f"{normals:,}")
    c3.metric("Attack", f"{attacks:,}")

    if not prev.empty:
        pc = prev["prediction"].value_counts().reset_index()
        pc.columns = ["prediction", "count"]
        fig = px.bar(
            pc, x="prediction", y="count", title="Sample: predicted class counts (first result rows only)"
        )
        st.plotly_chart(fig, use_container_width=True)

    if not prev.empty and "label" in prev.columns:
        try:
            y_true = prev["label"].astype(str).str.lower().map({"normal": 0, "attack": 1})
            y_pred = prev["prediction"].astype(str).str.lower().map({"normal": 0, "attack": 1})
            if y_true.notna().all() and y_pred.notna().all():
                cm_local = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(
                    cm_local,
                    index=["Actual Normal", "Actual Attack"],
                    columns=["Pred Normal", "Pred Attack"],
                )
                st.caption("Confusion matrix: **sample rows only** (not full 1GB).")
                st.plotly_chart(
                    px.imshow(cm_df, text_auto=True, title="Confusion matrix (preview sample)"),
                    use_container_width=True,
                )
        except Exception:
            pass

    if not prev.empty and "attack_probability" in prev.columns:
        st.plotly_chart(
            px.histogram(
                prev,
                x="attack_probability",
                nbins=40,
                title="Attack probability (preview sample only)",
            ),
            use_container_width=True,
        )

    if tpath and os.path.isfile(tpath):
        rsize = os.path.getsize(tpath)
        if rsize <= MAX_DOWNLOAD_RESULT_BYTES:
            st.download_button(
                "Download full results CSV",
                data=Path(tpath).read_bytes(),
                file_name="ids_results.csv",
                mime="text/csv",
            )
        else:
            st.caption("Full results CSV is large; offering **gzip** download (smaller than raw CSV).")
            tgz = Path(f"{tpath}.for_download.gz")
            if tgz.is_file():
                tgz.unlink()
            with open(tpath, "rb") as fi, gzip.open(tgz, "wb", compresslevel=6) as go:
                shutil.copyfileobj(fi, go, length=4 * 1024 * 1024)
            gzs = tgz.stat().st_size
            st.session_state["ids_pred_gz_path"] = str(tgz)
            cap = min(MAX_DOWNLOAD_RESULT_BYTES * 2, 200 * 1024 * 1024)
            if gzs <= cap:
                st.download_button(
                    "Download full results (gzip)",
                    data=tgz.read_bytes(),
                    file_name="ids_results.csv.gz",
                    mime="application/gzip",
                )
            else:
                st.warning(
                    "Even compressed, the file is very large for the browser. "
                    "Use **max rows** for a smaller run, or self-host. On-screen **metrics** are for the full scan."
                )

    st.session_state["last_scan_summary"] = {
        "rows": total,
        "normal": normals,
        "attack": attacks,
        "head_rows": prev.head(15).to_dict(orient="records") if not prev.empty else [],
        "has_labels": bool("label" in prev.columns) if not prev.empty else False,
    }


def render_upload_and_scan(pipe: Any) -> None:
    st.header("Upload Dataset to Scan Attacks")
    st.info(
        "**Large files:** The app can **scan in chunks** (no 1 GB RAM spike). "
        "**Slow “buffering” in the uploader** = your **browser is still uploading** to Streamlit — that part cannot be sped up in code. "
        "**Tip (Cloud):** use **Server file** after **Test case files → Generate** so **no 200 MB upload from your PC**. "
        "Or use **URL** so the **server downloads** the file. "
        "If you must upload from PC, **`.csv.gz`** is smaller. **Max rows** = faster tests. "
        "Charts on huge runs use a **preview sample**; metrics use **all rows**."
    )
    with st.expander("Juggad — instant scan (no upload, no file on disk)", expanded=True):
        st.caption(
            "Cloud pe **upload/Generate** fail ho to **yahi** use karo: benchmark jaisa synthetic data **RAM** mein banta hai, phir model chalta hai. "
            "Column `label` prediction se hata diya gaya."
        )
        jrows = st.selectbox(
            "Rows (keep ≤100k for free tier stability)",
            [2000, 5000, 10_000, 25_000, 50_000, 100_000],
            index=2,
            key="ids_juggad_rows",
        )
        if st.button("Run instant synthetic scan", type="primary", key="ids_juggad_run"):
            with st.status("Synthesizing + model…", expanded=True) as juggad_status:
                jdf = dataframe_nsl_synthetic(int(jrows), seed=42)
                juggad_status.write(f"Rows: {len(jdf):,} — running inference…")
                jout = predict_df(pipe, jdf, label_col="label")
                juggad_status.update(label="Done", state="complete")
            _write_scan_result_ui(jout, use_chunked=False)
            st.stop()

    demo_df = _load_default_demo_df()
    if demo_df is not None:
        st.success("Demo dataset found at `data/test.csv`. You can use it without uploading.")
        use_demo = st.toggle("Use demo dataset (`data/test.csv`)", value=True)
    else:
        use_demo = False

    data_source = "upload"
    up: Any = None
    work_path: Optional[Path] = None

    max_rows = 0
    if not use_demo:
        max_rows = int(
            st.number_input(
                "Max rows to load (0 = **entire file** — 200MB–1GB uses chunked mode automatically)",
                min_value=0,
                value=0,
                step=10_000,
                help="Use 0 to scan 100% of the data. Set e.g. 100000 to test faster.",
            )
        )
    nrows: Optional[int] = None if max_rows <= 0 else max_rows

    if not use_demo:
        data_source = st.radio(
            "Where is the CSV?",
            options=["server", "upload", "url"],
            format_func=lambda x: {
                "server": "Server file (fast on Cloud — no big PC upload)",
                "upload": "Upload from my PC (spinner = file still uploading)",
                "url": "From HTTPS URL (server downloads — no PC upload)",
            }[x],
            horizontal=True,
            key="ids_data_source",
        )
        if data_source != "url" and st.session_state.get("ids_url_temp_path"):
            _clear_url_temp_in_session()

        if data_source == "server":
            choices = _list_test_case_csvs()
            if choices:
                pick = st.selectbox(
                    "Pick a file already on the app server (`data/test_cases/`)",
                    options=[c.name for c in choices],
                    key="ids_server_pick",
                )
                work_path = TEST_CASES_DIR / pick
            else:
                st.warning(
                    "No CSV/GZ in `data/test_cases/` yet. Open **Test case files** → **Generate** "
                    "(e.g. 200 MB), then return here and choose **Server file** — **no long upload from your PC**."
                )
        elif data_source == "upload":
            st.warning(
                "The ⏳ next to the file name means **upload to Streamlit is still in progress** — this is **not** "
                "the ML scan yet. **200 MB** often takes **many minutes** on typical home upload speeds. "
                "To skip this: use **Server file** (generate on **Test case files**) or **URL**."
            )
            up = st.file_uploader(
                "Upload CSV or GZIP (.csv / .csv.gz) — up to 2 GB in this app config",
                type=["csv", "gz"],
                help="Gzip on your PC first for a smaller upload.",
            )
        else:
            st.caption(
                "The app downloads the file **on the server** (good for public links). "
                "Private networks / localhost URLs are blocked."
            )
            url_in = st.text_input("HTTPS URL to `.csv` or `.gz`", placeholder="https://…", key="ids_url_input")
            c_fetch, c_clear = st.columns(2)
            with c_fetch:
                if st.button("Fetch to server", type="primary", key="ids_url_fetch"):
                    if not (url_in or "").strip():
                        st.error("Paste a URL first.")
                    else:
                        ok, _m = _url_host_is_public(url_in)
                        if not ok:
                            st.error(_m)
                        else:
                            with st.status("Downloading…", expanded=True) as dl_st:
                                pth, err = _download_url_to_tempfile(url_in)
                                if err:
                                    dl_st.update(label="Download failed", state="error")
                                    st.error(err)
                                else:
                                    st.session_state["ids_url_temp_path"] = pth
                                    dl_st.update(label="Download complete", state="complete")
                                    st.rerun()
            with c_clear:
                if st.button("Clear fetched file", key="ids_url_clear"):
                    _clear_url_temp_in_session()
                    st.rerun()
            _ut = st.session_state.get("ids_url_temp_path")
            if isinstance(_ut, str) and os.path.isfile(_ut):
                work_path = Path(_ut)

    label_col = st.text_input("Label column (optional, will be ignored)", value="")
    lcol: Optional[str] = label_col.strip() or None

    if use_demo:
        run_ok = demo_df is not None
    elif data_source == "server":
        run_ok = work_path is not None and work_path.is_file()
    elif data_source == "url":
        run_ok = work_path is not None and work_path.is_file()
    else:
        run_ok = up is not None

    st.subheader("Preview (first 100 rows)")
    if use_demo and demo_df is not None:
        st.dataframe(demo_df.head(100), use_container_width=True)
        st.caption("Demo set — first 100 rows.")
    elif not use_demo and not run_ok:
        st.caption("Select a **server file**, finish **URL fetch**, or finish **PC upload** to see a preview.")
    elif work_path is not None and work_path.is_file():
        try:
            prev = _read_path_csv(work_path, nrows=100)
            st.dataframe(prev, use_container_width=True)
            st.caption(
                f"On-server file ≈ **{_fmt_file_size(work_path.stat().st_size)}** — **not** uploading from your browser."
            )
        except Exception as e:
            st.error(f"Preview failed: {e}")
    elif up is not None:
        prev = _preview_upload_nrows(up, 100)
        sz = _uploaded_file_size_bytes(up)
        st.dataframe(prev, use_container_width=True)
        st.caption(
            f"Uploaded size ≈ **{_fmt_file_size(sz)}**. First 100 rows. "
            "Wait until the uploader spinner finishes, then click **Run Detection**."
        )
    else:
        st.warning("Turn on **demo**, or choose a data source above.")

    if not st.button("Run Detection", type="primary", disabled=not run_ok):
        return

    if use_demo and demo_df is None:
        st.error("Demo dataset not found. Place `data/test.csv` or switch off the demo toggle and upload a file.")
        return

    if use_demo and demo_df is not None:
        with st.spinner("Running detection on full demo data…"):
            out = predict_df(pipe, demo_df, label_col=lcol)
        _write_scan_result_ui(out, use_chunked=False)
        return

    if work_path is not None and work_path.is_file():
        if nrows is not None and nrows > 0:
            with st.spinner(f"Loading first {nrows:,} rows…"):
                df = _read_path_csv(work_path, nrows=nrows)
            out = predict_df(pipe, df, label_col=lcol)
            _write_scan_result_ui(out, use_chunked=False)
            return

        sz2 = work_path.stat().st_size
        if sz2 > LARGE_UPLOAD_BYTES and (nrows is None or nrows <= 0):
            _cleanup_pred_temp_keys()
            st.success(
                f"**Large file mode** (~{_fmt_file_size(sz2)}): processing in **chunks** (safe for 1 GB+). "
                "This can take many minutes; do not close the tab."
            )
            with st.status("Scanning file in chunks (full dataset)…", expanded=True) as status:
                summary, tpath, prev, err = predict_from_path_csv_in_chunks(
                    pipe, str(work_path), label_col=lcol, chunksize=200_000
                )
                if err:
                    status.update(label="Scan failed", state="error")
                    st.error(err)
                    return
                status.update(label="Scan complete", state="complete")
            if tpath and os.path.isfile(tpath):
                st.session_state["ids_pred_temp_csv"] = tpath
            _write_scan_result_chunked_ui(summary, prev, tpath, lcol)
            return

        with st.spinner("Loading file…"):
            df = _read_path_csv(work_path, nrows=None)
        out = predict_df(pipe, df, label_col=lcol)
        _write_scan_result_ui(out, use_chunked=False)
        return

    if up is None:
        return

    if nrows is not None and nrows > 0:
        up.seek(0)
        with st.spinner(f"Loading first {nrows:,} rows…"):
            df = _read_streamlit_uploaded_csv(up, nrows=nrows)
        out = predict_df(pipe, df, label_col=lcol)
        _write_scan_result_ui(out, use_chunked=False)
        return

    up.seek(0)
    sz2 = _uploaded_file_size_bytes(up)
    if sz2 > LARGE_UPLOAD_BYTES and (nrows is None or nrows <= 0):
        _cleanup_pred_temp_keys()
        st.success(
            f"**Large file mode** (~{_fmt_file_size(sz2)}): processing in **chunks** (safe for 1 GB+). "
            "This can take many minutes; do not close the tab."
        )
        up.seek(0)
        with st.status("Scanning file in chunks (full dataset)…", expanded=True) as status:
            summary, tpath, prev, err = predict_from_uploaded_csv_in_chunks(
                pipe, up, label_col=lcol, chunksize=200_000
            )
            if err:
                status.update(label="Scan failed", state="error")
                st.error(err)
                return
            status.update(label="Scan complete", state="complete")
        if tpath and os.path.isfile(tpath):
            st.session_state["ids_pred_temp_csv"] = tpath
        _write_scan_result_chunked_ui(summary, prev, tpath, lcol)
    else:
        with st.spinner("Loading file…"):
            up.seek(0)
            df = _read_streamlit_uploaded_csv(up, nrows=None)
        out = predict_df(pipe, df, label_col=lcol)
        _write_scan_result_ui(out, use_chunked=False)


def render_test_case_library() -> None:
    """`data/test_cases/`: 8+ benchmark sizes; upload (≤ limit) or generate (any size on server)."""
    st.header("Test case files (benchmark sizes)")
    rel = TEST_CASES_DIR.relative_to(PROJECT_ROOT)
    st.caption(
        f"Folder `{rel}/` — **9 preset sizes** (10 KB → 1 GB). NSL-KDD–shaped synthetic CSV; "
        "use **Upload & Scan** to run the model on one of these files. "
        "**Generate** runs **in-process** (no extra Python subprocess) so it works more reliably on Streamlit Cloud. "
        f"**On this page, browser upload is only for tiers ≤ ~{_fmt_file_size(TEST_CASE_IN_PAGE_UPLOAD_MAX)}.** "
        "For **100 MB+** tiers we **turn off upload** (it only caused endless buffering) — use **Generate this file only** instead. "
        f"**Max file size in app config:** **{MAX_UPLOAD_MB} MB** (`.streamlit/config.toml` `maxUploadSize`). "
        "After **Generate**, open **Upload & Scan → Server file** to scan with **no** big upload from your PC."
    )
    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate all 8 (10 KB → 200 MB)", type="primary", key="tc_bench8"):
            with st.status("Writing 8 benchmark CSVs in-process (may take a few minutes)…", expanded=True) as status:
                failed: list[str] = []
                for _label, fname, tbytes in BENCHMARK_FILES[:8]:
                    st.write(f"… `{fname}`")
                    dest = TEST_CASES_DIR / fname
                    ok, err = _generate_test_case_file_inprocess(dest, tbytes)
                    if not ok:
                        failed.append(f"{fname}: {err}")
                if failed:
                    status.update(label="Some writes failed", state="error")
                    st.error("\n".join(failed))
                else:
                    status.update(label="Done.", state="complete")
                    st.rerun()
    with c2:
        if st.button("Generate + ~1 GB file (very slow)", key="tc_bench1g"):
            with st.status("Writing 9 files including ~1 GB (long, may time out on free Cloud)…", expanded=True) as status:
                failed = []
                for _label, fname, tbytes in BENCHMARK_FILES:
                    st.write(f"… `{fname}`")
                    dest = TEST_CASES_DIR / fname
                    ok, err = _generate_test_case_file_inprocess(dest, tbytes)
                    if not ok:
                        failed.append(f"{fname}: {err}")
                if failed:
                    status.update(label="Some writes failed", state="error")
                    st.error("\n".join(failed))
                else:
                    status.update(label="Done.", state="complete")
                    st.rerun()

    for label, fname, tbytes in BENCHMARK_FILES:
        path = TEST_CASES_DIR / fname
        exists = path.is_file()
        upload_ok = tbytes <= MAX_UPLOAD_MB * 1024 * 1024
        allow_in_page_upload = upload_ok and tbytes <= TEST_CASE_IN_PAGE_UPLOAD_MAX
        hint = (
            f"Target on disk ≈ {label} — one file, grows by rows until size reached. "
            + ("Fits under upload cap." if upload_ok else f"Larger than {MAX_UPLOAD_MB} MB cap — use Generate, not upload.")
        )
        with st.expander(f"{label} — `{fname}`", expanded=False):
            st.caption(hint)
            if exists:
                st.success(f"Found — **{_fmt_file_size(path.stat().st_size)}**")
            else:
                st.info("Missing — use **Generate this file only** (recommended for 100MB+), or small-tier **upload** where enabled.")
            if upload_ok and not allow_in_page_upload:
                st.warning(
                    f"**Upload is disabled** for this tier: putting **{_fmt_file_size(tbytes)}** through the "
                    "browser uploader just **buffers 10–40+ min** and looks broken. "
                    "Use **Generate this file only** (writes on the server), then go to **Upload & Scan → Server file**."
                )
            fup = st.file_uploader(
                "Upload CSV"
                + (
                    ""
                    if allow_in_page_upload
                    else (
                        f" (off for this tier — use **Generate**; max in-page upload size is ~{_fmt_file_size(TEST_CASE_IN_PAGE_UPLOAD_MAX)})"
                    )
                ),
                type=["csv"],
                key=f"tc_up_{fname}",
                disabled=not allow_in_page_upload,
            )
            if allow_in_page_upload and fup is not None and st.button("Save as " + fname, key=f"tc_save_{fname}"):
                try:
                    _save_upload_to_test_cases(fup, fname)
                    st.success("Saved.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if st.button("Generate this file only", key=f"tc_gen_{fname}"):
                with st.status(f"Writing `{fname}` (in-process)…", expanded=True) as status:
                    ok, err = _generate_test_case_file_inprocess(path, tbytes)
                    if not ok:
                        status.update(label="Failed.", state="error")
                        st.error(err or "Write failed.")
                    else:
                        status.update(label="Done.", state="complete")
                        st.rerun()

    st.divider()
    st.subheader("All CSVs in this folder")
    all_csv = sorted(TEST_CASES_DIR.glob("*.csv"), key=lambda p: p.name.lower())
    if not all_csv:
        st.caption("No `.csv` files yet.")
    else:
        for p in all_csv:
            st.write(f"- `{p.name}` — {_fmt_file_size(p.stat().st_size)}")


def render_live_monitoring(pipe: Any) -> None:
    st.header("Live Attack Monitoring")
    st.caption("Default mode is simulation (safe). Packet capture requires root + scapy.")

    mode = st.radio(
        "Live mode",
        options=["Simulation"],
        index=0,
        horizontal=True,
    )

    rate = st.slider("Events/sec", min_value=1, max_value=10, value=2)
    st.info(f"Scapy available: {scapy_sniff_available()}")

    demo_df = _load_default_demo_df()
    if demo_df is not None:
        use_demo = st.toggle("Use demo dataset schema (`data/test.csv`)", value=True, key="live_demo")
    else:
        use_demo = False

    schema_up = None if use_demo else st.file_uploader(
        "Upload CSV or GZIP for schema (only first rows are read)",
        type=["csv", "gz"],
        key="schema",
    )
    if not use_demo and schema_up is None:
        st.warning("Upload any CSV (same columns as your training data) to start simulation.")
        return

    schema_df = (
        demo_df
        if use_demo
        else _read_streamlit_uploaded_csv(schema_up, nrows=5000)  # type: ignore[arg-type]
    ).head(500)
    placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.button("Start Live Stream"):
        events = []
        gen = simulate_stream(pipe, schema_df=schema_df, rate_per_sec=float(rate))
        for _ in range(200):  # bounded loop for Streamlit run
            ev = next(gen)
            events.append(
                {
                    "ts": ev.ts,
                    "prediction": ev.prediction,
                    "attack_probability": ev.attack_probability,
                }
            )
            df_ev = pd.DataFrame(events)
            placeholder.dataframe(df_ev.tail(20), use_container_width=True)
            fig = px.line(df_ev, x="ts", y="attack_probability", color="prediction", title="Live attack probability")
            chart_placeholder.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not check_login():
        return

    if st.sidebar.button("Logout"):
        st.session_state.ids_logout_clears_cookie = True
        st.session_state.authed = False
        st.session_state.user_email = ""
        _emit_clear_session_cookie_javascript()
        for k in ("last_report_pdf", "last_scan_summary"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    em = str(st.session_state.get("user_email", "") or "").strip()
    if em:
        st.sidebar.caption(f"Signed in as **{em}**")

    model_path = sidebar_model_picker()
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=["Home", "Upload & Scan", "Live Monitoring", "Test case files"],
        index=0,
    )

    if not Path(model_path).exists():
        st.title(APP_TITLE)
        maybe_bootstrap_demo_assets(model_path)
        st.code("Manual train: python -m ids.train --data data/train.csv --label-col label --out models/ids_model.joblib")
        return

    pipe = load_model(model_path)

    st.title(APP_TITLE)

    if page == "Home":
        render_metrics()
    elif page == "Upload & Scan":
        render_upload_and_scan(pipe)
    elif page == "Live Monitoring":
        render_live_monitoring(pipe)
    elif page == "Test case files":
        render_test_case_library()


if __name__ == "__main__":
    main()

