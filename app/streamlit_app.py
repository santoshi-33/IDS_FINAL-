from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix
import subprocess

# Ensure project root is importable when running via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ids.live import scapy_sniff_available, simulate_stream
from ids.pipeline import predict_df
from ids.reporting import build_ids_report_pdf, load_metrics_json
from ids.user_store import env_bootstrap_exists, sign_up, verify_user


APP_TITLE = "ML-based Intrusion Detection System (IDS)"

# Benchmark CSVs: scripts/generate_test_datasets.py (--benchmark-tiers, --out-file, …)
TEST_CASES_DIR = PROJECT_ROOT / "data" / "test_cases"
# Must match .streamlit/config.toml [server] maxUploadSize (MB) for local/self-host
MAX_UPLOAD_MB = 2048
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


def _run_generate_subprocess(extra: list[str]) -> subprocess.CompletedProcess[str]:
    script = PROJECT_ROOT / "scripts" / "generate_test_datasets.py"
    cmd = [sys.executable, str(script), "--out-dir", "data/test_cases", "--no-legacy-synth-names", *extra]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)


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


def check_login() -> bool:
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""

    if st.session_state.authed:
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
                        st.session_state.authed = True
                        st.session_state.user_email = lg_email
                        st.rerun()
                    else:
                        st.error(msg)

    st.caption(
        "Accounts are stored locally in `data/app_users.json` (hashed passwords). "
        "On Streamlit Community Cloud, this file may reset when the app rebuilds—"
        "use Sign up again, or rely on optional `IDS_USER`/`IDS_PASS` secrets for a fixed login."
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


def render_upload_and_scan(pipe: Any) -> None:
    st.header("Upload Dataset to Scan Attacks")
    demo_df = _load_default_demo_df()
    if demo_df is not None:
        st.success("Demo dataset found at `data/test.csv`. You can use it without uploading.")
        use_demo = st.toggle("Use demo dataset (`data/test.csv`)", value=True)
    else:
        use_demo = False

    up = None if use_demo else st.file_uploader("Upload CSV", type=["csv"])
    label_col = st.text_input("Label column (optional, will be ignored)", value="")

    if not use_demo and up is None:
        st.caption("Upload a CSV similar to NSL-KDD / CICIDS2017 feature tables.")
        return

    df = demo_df if use_demo else pd.read_csv(up)  # type: ignore[arg-type]
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    if st.button("Run Detection"):
        out = predict_df(pipe, df, label_col=(label_col.strip() or None))

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
        c1.metric("Total rows", f"{total}")
        c2.metric("Normal", f"{normals}")
        c3.metric("Attack", f"{attacks}")

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

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="ids_results.csv",
            mime="text/csv",
        )

        # Save summary for PDF report
        st.session_state["last_scan_summary"] = {
            "rows": int(len(out)),
            "normal": int((out["prediction"] == "normal").sum()),
            "attack": int((out["prediction"] == "attack").sum()),
            "head_rows": out.head(15).to_dict(orient="records"),
            "has_labels": bool("label" in out.columns),
        }


def render_test_case_library() -> None:
    """`data/test_cases/`: 8+ benchmark sizes; upload (≤ limit) or generate (any size on server)."""
    st.header("Test case files (benchmark sizes)")
    rel = TEST_CASES_DIR.relative_to(PROJECT_ROOT)
    st.caption(
        f"Folder `{rel}/` — **9 preset sizes** (10 KB → 1 GB). NSL-KDD–shaped synthetic CSV; "
        "use **Upload & Scan** to run the model on one of these files. "
        f"**Max upload in this app config:** **{MAX_UPLOAD_MB} MB** (`.streamlit/config.toml` `maxUploadSize`). "
        "On Streamlit Cloud the host may cap lower — for big files, use **Generate** on the server, not upload."
    )
    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate all 8 (10 KB → 200 MB)", type="primary", key="tc_bench8"):
            with st.status("Generating 8 benchmark CSVs (may take several minutes)…", expanded=True) as status:
                proc = _run_generate_subprocess(["--benchmark-tiers"])
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                st.code(out or "(no output)")
                if proc.returncode == 0:
                    status.update(label="Done.", state="complete")
                    st.rerun()
                else:
                    status.update(label="Failed.", state="error")
                    st.error("Generator failed.")
    with c2:
        if st.button("Generate + ~1 GB file (very slow)", key="tc_bench1g"):
            with st.status("Generating 8 tiers + ~1 GB (long)…", expanded=True) as status:
                proc = _run_generate_subprocess(["--benchmark-tiers", "--include-1gb"])
                st.code((proc.stdout or "") + "\n" + (proc.stderr or "") or "(no output)")
                if proc.returncode == 0:
                    status.update(label="Done.", state="complete")
                    st.rerun()
                else:
                    status.update(label="Failed.", state="error")
                    st.error("Generator failed.")

    for label, fname, tbytes in BENCHMARK_FILES:
        gen_args: list[str] = [
            "--out-file",
            fname,
            "--target-bytes",
            str(tbytes),
        ]
        path = TEST_CASES_DIR / fname
        exists = path.is_file()
        upload_ok = tbytes <= MAX_UPLOAD_MB * 1024 * 1024
        hint = (
            f"Target on disk ≈ {label} — one file, grows by rows until size reached. "
            + ("Fits under upload cap." if upload_ok else f"Larger than {MAX_UPLOAD_MB} MB cap — use Generate, not upload.")
        )
        with st.expander(f"{label} — `{fname}`", expanded=False):
            st.caption(hint)
            if exists:
                st.success(f"Found — **{_fmt_file_size(path.stat().st_size)}**")
            else:
                st.info("Missing — **Generate** here or **upload** if under max size.")
            fup = st.file_uploader(
                "Upload CSV" + ("" if upload_ok else f" (file too big for {MAX_UPLOAD_MB} MB cap — generate instead)"),
                type=["csv"],
                key=f"tc_up_{fname}",
                disabled=not upload_ok,
            )
            if upload_ok and fup is not None and st.button("Save as " + fname, key=f"tc_save_{fname}"):
                try:
                    _save_upload_to_test_cases(fup, fname)
                    st.success("Saved.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
            if st.button("Generate this file only", key=f"tc_gen_{fname}"):
                with st.status(f"Writing `{fname}`…", expanded=True) as status:
                    proc = _run_generate_subprocess(gen_args)
                    st.code((proc.stdout or "") + "\n" + (proc.stderr or "") or "(no output)")
                    if proc.returncode == 0:
                        status.update(label="Done.", state="complete")
                        st.rerun()
                    else:
                        status.update(label="Failed.", state="error")
                        st.error("Generator failed.")

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
        "Upload a CSV to infer schema for simulation", type=["csv"], key="schema"
    )
    if not use_demo and schema_up is None:
        st.warning("Upload any CSV (same columns as your training data) to start simulation.")
        return

    schema_df = (demo_df if use_demo else pd.read_csv(schema_up)).head(500)  # type: ignore[arg-type]
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
        st.session_state.authed = False
        st.session_state.user_email = ""
        for k in ("last_report_pdf", "last_scan_summary"):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

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

