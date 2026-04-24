from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore

    _HAS_XGBOOST = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGBOOST = False

from .data import basic_clean, normalize_label_series


@dataclass(frozen=True)
class TrainArtifacts:
    pipeline: Any
    feature_columns: List[str]
    label_col: str


def _infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = _infer_column_types(X)

    numeric_pipe = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    X: pd.DataFrame,
    *,
    use_smote: bool = True,
    use_pca: bool = False,
    pca_components: int = 20,
    random_state: int = 42,
) -> ImbPipeline:
    pre = build_preprocessor(X)

    steps: List[Tuple[str, Any]] = [("preprocess", pre)]
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components, random_state=random_state)))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    if _HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        # Fallback when xgboost isn't available (keeps project runnable in restricted envs).
        model = HistGradientBoostingClassifier(random_state=random_state)
    steps.append(("model", model))
    return ImbPipeline(steps=steps)


def prepare_xy(df: pd.DataFrame, *, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = basic_clean(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing in dataset.")
    y = normalize_label_series(df[label_col])
    X = df.drop(columns=[label_col])
    return X, y


def train_eval(
    df: pd.DataFrame,
    *,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True,
    use_pca: bool = False,
    pca_components: int = 20,
) -> Tuple[TrainArtifacts, Dict[str, Any]]:
    X, y = prepare_xy(df, label_col=label_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = build_pipeline(
        X_train,
        use_smote=use_smote,
        use_pca=use_pca,
        pca_components=pca_components,
        random_state=random_state,
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = None
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc) if roc is not None else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_col": label_col,
        "use_smote": bool(use_smote),
        "use_pca": bool(use_pca),
        "pca_components": int(pca_components),
    }

    artifacts = TrainArtifacts(
        pipeline=pipe, feature_columns=X.columns.tolist(), label_col=label_col
    )
    return artifacts, metrics


def predict_df(
    pipeline: Any,
    df: pd.DataFrame,
    *,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    df = basic_clean(df)
    X = df.drop(columns=[label_col]) if (label_col and label_col in df.columns) else df

    pred = pipeline.predict(X)
    proba = None
    try:
        proba = pipeline.predict_proba(X)[:, 1]
    except Exception:
        proba = None

    out = df.copy()
    out["prediction"] = np.where(pred == 1, "attack", "normal")
    if proba is not None:
        out["attack_probability"] = proba
    return out


# Use when uploaded CSV is hundreds of MB–1+ GB: avoid holding the whole file in memory at once.
LARGE_CSV_DEFAULT_CHUNK = 200_000

_STREAM_COMPRESSION = {".gz": "gzip"}


def _read_csv_kwargs_for_name(name: str) -> dict:
    n = (name or "").lower()
    for suf, c in _STREAM_COMPRESSION.items():
        if n.endswith(suf):
            return {"compression": c}
    return {}


def predict_from_uploaded_csv_in_chunks(
    pipeline: Any,
    file_obj: Any,
    *,
    label_col: Optional[str] = None,
    chunksize: int = LARGE_CSV_DEFAULT_CHUNK,
) -> tuple[dict, str, pd.DataFrame, Optional[str]]:
    """
    Read CSV in chunks, predict per chunk, append results to a temp file.

    Returns
    -------
    summary : dict
        ``rows``, ``attack``, ``normal``, optional label stats
    out_path : str
        Path to full results CSV (caller should ``os.remove`` when done, if desired).
    preview : DataFrame
        A few hundred rows to display (head + more from first chunks).
    err : optional str
        Set if a non-fatal issue occurred
    """
    import os
    import tempfile

    name = str(getattr(file_obj, "name", "") or "")

    tfp = tempfile.NamedTemporaryFile(
        delete=False, prefix="ids_pred_", suffix=".csv", mode="w", encoding="utf-8", newline=""
    )
    out_path = tfp.name
    tfp.close()

    kwargs = {"chunksize": int(chunksize), "low_memory": False, **_read_csv_kwargs_for_name(name)}

    total = 0
    attack = 0
    preview_rows: list[pd.DataFrame] = []
    max_preview = 1_200
    n_preview = 0
    first_chunk = True

    reader = pd.read_csv(file_obj, **kwargs)
    for chunk in reader:
        if not isinstance(chunk, pd.DataFrame) or len(chunk) == 0:
            break
        out = predict_df(pipeline, chunk, label_col=label_col)
        t = len(out)
        total += t
        attack += int((out["prediction"] == "attack").sum())
        if first_chunk:
            out.to_csv(out_path, index=False, mode="w", header=True)
            first_chunk = False
        else:
            out.to_csv(out_path, index=False, mode="a", header=False)
        if n_preview < max_preview:
            take = min(len(out), max_preview - n_preview)
            if take:
                preview_rows.append(out.head(take))
                n_preview += take
    normal = int(total) - int(attack)
    if preview_rows:
        prev = pd.concat(preview_rows, ignore_index=True)
    else:
        prev = pd.DataFrame()
    err = None
    if first_chunk:
        err = "No rows read from the file."
        try:
            os.remove(out_path)
        except OSError:
            pass
        return ({"rows": 0, "attack": 0, "normal": 0, "result_path": ""}, "", prev, err)
    summary = {
        "rows": int(total),
        "attack": int(attack),
        "normal": int(normal),
        "result_path": out_path,
    }
    return summary, out_path, prev, err

