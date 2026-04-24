from __future__ import annotations

import io
import json
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF


def build_ids_report_pdf(
    *,
    title: str,
    user_email: str,
    metrics: Optional[Dict[str, Any]],
    scan_summary: Optional[Dict[str, Any]],
) -> bytes:
    """Return PDF bytes for the IDS report."""

    class ReportPDF(FPDF):
        def header(self) -> None:
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(2)

        def footer(self) -> None:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Executive summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, f"Generated: {now}\nUser: {user_email}")

    if metrics:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Model training metrics (hold-out test)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        lines = [
            f"Accuracy: {metrics.get('accuracy')}",
            f"Precision: {metrics.get('precision')}",
            f"Recall: {metrics.get('recall')}",
            f"F1: {metrics.get('f1')}",
            f"ROC-AUC: {metrics.get('roc_auc')}",
        ]
        pdf.multi_cell(0, 5, "\n".join([str(x) for x in lines if x is not None]))

        cm = metrics.get("confusion_matrix")
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Confusion matrix (test split)", new_x="LMARGIN", new_y="NEXT")
            _render_table(
                pdf,
                headers=["", "Pred Normal", "Pred Attack"],
                rows=[
                    ["Actual Normal", str(cm[0][0]), str(cm[0][1])],
                    ["Actual Attack", str(cm[1][0]), str(cm[1][1])],
                ],
            )

    if scan_summary:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Latest scan (Upload & Run Detection)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for k, v in scan_summary.items():
            if k in ("head_rows", "confusion_matrix"):
                continue
            pdf.multi_cell(0, 5, f"{k}: {v}")
        if scan_summary.get("head_rows"):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Sample rows (first 15)", new_x="LMARGIN", new_y="NEXT")
            _render_df_table(pdf, scan_summary["head_rows"][:15])

    out = io.BytesIO()
    pdf_data = pdf.output()
    if isinstance(pdf_data, (bytes, bytearray)):
        out.write(pdf_data)
    else:
        out.write(str(pdf_data).encode("utf-8"))
    return out.getvalue()


def _render_table(pdf: FPDF, headers: List[str], rows: List[List[str]]) -> None:
    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / max(len(headers), 1)
    pdf.set_font("Helvetica", "B", 9)
    for h in headers:
        pdf.cell(col_w, 7, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=9)
    for r in rows:
        for c in r:
            pdf.cell(col_w, 7, str(c), border=1)
        pdf.ln()


def _render_df_table(pdf: FPDF, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols: List[str] = list(rows[0].keys())[:10]
    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / max(len(cols), 1)
    pdf.set_font("Helvetica", "B", 7)
    for c in cols:
        pdf.cell(col_w, 6, c[:20], border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=7)
    for r in rows:
        for c in cols:
            v = r.get(c, "")
            s = str(v)[:20]
            pdf.cell(col_w, 6, s, border=1)
        pdf.ln()


def load_metrics_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def try_send_email_with_pdf(
    *,
    to_email: str,
    subject: str,
    body: str,
    pdf_bytes: bytes,
    filename: str = "ids_report.pdf",
) -> None:
    smtp_host = (os.getenv("IDS_SMTP_HOST") or "").strip()
    smtp_port = int(os.getenv("IDS_SMTP_PORT") or "587")
    smtp_user = (os.getenv("IDS_SMTP_USER") or "").strip()
    smtp_pass = (os.getenv("IDS_SMTP_PASS") or "").strip()
    from_addr = (os.getenv("IDS_SMTP_FROM") or smtp_user or "").strip()
    ssl_mode = (os.getenv("IDS_SMTP_SSL") or "").strip().lower() in ("1", "true", "yes", "y")
    starttls_default = (os.getenv("IDS_SMTP_STARTTLS", "1").strip() not in ("0", "false", "False"))

    # Brevo commonly: 587 + STARTTLS, or 465 + implicit TLS (SMTP_SSL)
    if smtp_port == 465:
        use_smtp_ssl = True
        use_starttls = False
    else:
        use_smtp_ssl = ssl_mode
        use_starttls = starttls_default and not use_smtp_ssl

    if not smtp_host or not from_addr or not smtp_user or not smtp_pass:
        raise RuntimeError(
            "SMTP not configured. Set IDS_SMTP_HOST, IDS_SMTP_PORT, IDS_SMTP_USER, "
            "IDS_SMTP_PASS, IDS_SMTP_FROM (and optionally IDS_SMTP_STARTTLS=0) in environment/secrets."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_email
    msg.set_content(body)
    msg.add_attachment(
        pdf_bytes, maintype="application", subtype="pdf", filename=filename
    )

    context = ssl.create_default_context()
    if use_smtp_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if use_starttls:
                server.starttls(context=context)
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
