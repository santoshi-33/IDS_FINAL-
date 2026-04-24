"""
NSL-KDD–shaped synthetic tabular data (same as scripts/generate_test_datasets.py).
Shared by the CLI script and Streamlit (in-process generate + instant demo scan).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Columns must match training CSV (no attack_type; label is last)
COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

PROTO = ["tcp", "udp", "icmp"]
SERVICE = [
    "http",
    "https",
    "ftp_data",
    "smtp",
    "private",
    "other",
    "domain",
    "ecr_i",
    "auth",
]
FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "RSTR", "S1", "S2", "S3", "OTH"]


def random_nsl_kdd_row_lists(rng: np.random.Generator, n: int) -> List[List[object]]:
    """Return ``n`` rows as lists (same as legacy ``_random_rows``)."""
    rows: List[List[object]] = []
    for _ in range(n):
        proto = str(rng.choice(PROTO))
        service = str(rng.choice(SERVICE))
        flag = str(rng.choice(FLAGS))
        src_b = int(rng.integers(0, 100_000))
        dst_b = int(rng.integers(0, 1_000_000))
        land = int(rng.integers(0, 2))
        wrong_fragment = int(rng.integers(0, 4))
        urgent = int(rng.integers(0, 2))
        hot = int(rng.integers(0, 20))
        num_failed = int(rng.integers(0, 5))
        logged_in = int(rng.integers(0, 2))
        num_comp = int(rng.integers(0, 10))
        root_sh = int(rng.integers(0, 2))
        su_a = int(rng.integers(0, 2))
        num_root = int(rng.integers(0, 5))
        nfc = int(rng.integers(0, 10))
        nshells = int(rng.integers(0, 5))
        naf = int(rng.integers(0, 20))
        noc = 0
        is_host = int(rng.integers(0, 2))
        is_guest = int(rng.integers(0, 2))
        count = int(rng.integers(0, 500))
        srv_count = int(rng.integers(0, 500))
        rate_block_a = [float(rng.random()) for _ in range(7)]
        dst_hc = int(rng.integers(0, 300))
        dst_hsvc = int(rng.integers(0, 300))
        rate_block_b = [float(rng.random()) for _ in range(8)]
        label = "normal" if rng.random() < 0.5 else "attack"
        row = [
            int(rng.integers(0, 30_000)),
            proto,
            service,
            flag,
            src_b,
            dst_b,
            land,
            wrong_fragment,
            urgent,
            hot,
            num_failed,
            logged_in,
            num_comp,
            root_sh,
            su_a,
            num_root,
            nfc,
            nshells,
            naf,
            noc,
            is_host,
            is_guest,
            count,
            srv_count,
        ]
        row.extend(rate_block_a)
        row.extend([dst_hc, dst_hsvc])
        row.extend(rate_block_b)
        row.append(label)
        rows.append(row)
    return rows


def write_csv(path: Path, n_rows: int, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        chunk = 5_000
        left = n_rows
        while left > 0:
            take = min(chunk, left)
            for row in random_nsl_kdd_row_lists(rng, take):
                w.writerow(row)
            left -= take


def write_until_size(
    path: Path, target_bytes: int, rng: np.random.Generator, chunk: int = 5_000
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        f.flush()
        while f.tell() < target_bytes:
            remain = target_bytes - f.tell()
            n_batch = chunk if remain > 120_000 else max(1, min(chunk, remain // 150 + 1))
            for row in random_nsl_kdd_row_lists(rng, n_batch):
                w.writerow(row)
                if f.tell() >= target_bytes:
                    break
            f.flush()
        total = f.tell()
        gb = total / (1024**3)
        if gb >= 0.1:
            print(f"  ... {path.name} ~{gb:.2f} GB", flush=True)
        else:
            print(f"  ... {path.name} ~{total / (1024**2):.1f} MB", flush=True)


def dataframe_nsl_synthetic(n_rows: int, *, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame in memory (for instant demo scans; do not use for multi‑GB)."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if n_rows == 0:
        return pd.DataFrame(columns=COLUMNS)
    if n_rows > 1_000_000:
        raise ValueError("Use a value ≤ 1,000,000 for this in-memory path.")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(random_nsl_kdd_row_lists(rng, n_rows), columns=COLUMNS)
