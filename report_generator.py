# report_generator.py
"""
BitStack - FAST processing mode:
- cleaning, EDA (distributions, corr heatmap, summary statistics)
- trains several candidate models on the CLEANED dataset
- FAST defaults: sampling + small trees for speed
- saves artifacts into outdir and returns training_info for the caller
"""

import matplotlib
matplotlib.use("Agg")

from __future__ import annotations
import os
import traceback
import base64
import pickle
import datetime
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.metrics import accuracy_score, r2_score

sns.set(style="darkgrid")
plt.rcParams["figure.dpi"] = 100


# ---------- helpers ----------
def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def onehot_compat():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------- EDA helpers ----------
def detect_outliers(df: pd.DataFrame, num_cols: List[str]):
    out = {}
    for c in num_cols:
        if c not in df.columns or df[c].nunique() < 2:
            out[c] = 0
            continue
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[c] = int(((df[c] < low) | (df[c] > high)).sum())
    return out


def detect_skewness(df: pd.DataFrame, num_cols: List[str]):
    if not num_cols:
        return pd.Series(dtype=float)
    skew = df[num_cols].skew()
    return skew[abs(skew) > 1]


def detect_low_variance(df: pd.DataFrame, num_cols: List[str]):
    if not num_cols:
        return pd.Series(dtype=float)
    var = df[num_cols].var()
    return var[var < 0.01]


def detect_high_corr(df: pd.DataFrame, num_cols: List[str], thr: float = 0.5):
    if not num_cols or len(num_cols) < 2:
        return pd.Series(dtype=float)
    corr = df[num_cols].corr()
    pairs = corr.unstack()
    pairs = pairs[
        (pairs.index.get_level_values(0) != pairs.index.get_level_values(1))
    ]
    pairs = pairs[abs(pairs) > thr]
    pairs = pairs[pairs.index.map(lambda t: t[0] < t[1])]
    return pairs.sort_values(ascending=False)


# ---------- target heuristics ----------
def find_target_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit in df.columns:
            return explicit
        for c in df.columns:
            if c.lower() == explicit.lower():
                return c
        return None

    commons = ["target", "label", "y", "class", "outcome", "survived", "response"]
    for p in commons:
        for c in df.columns:
            if c.lower() == p:
                return c
    for p in commons:
        for c in df.columns:
            if p in str(c).lower():
                return c

    # prefer low-unique columns (<=20 unique)
    n = df.shape[0]
    candidates = []
    for c in df.columns:
        if str(c).lower().startswith("unnamed"):
            continue
        nunique = df[c].nunique(dropna=True)
        if 1 < nunique <= 20:
            candidates.append((c, nunique))
    if candidates:
        candidates = sorted(candidates, key=lambda x: x[1])
        return candidates[0][0]

    # fallback last column if seems valid
    last = df.columns[-1]
    if not df[last].isna().all() and df[last].nunique(dropna=True) > 1:
        return last
    return None


def infer_task_type(series: pd.Series) -> str:
    if series is None or len(series) == 0:
        return "regression"
    if series.dtype.kind in "O" or str(series.dtype).startswith("category"):
        return "classification"
    nunique = int(series.nunique(dropna=True))
    ratio = nunique / max(1, len(series))
    if nunique <= 20 or ratio < 0.05:
        return "classification"
    return "regression"


# ---------- cleaning & artifact creation ----------
def create_cleaned_full(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfc = df.copy()
    # drop completely empty columns
    for c in list(dfc.columns):
        if dfc[c].isna().all():
            dfc.drop(columns=[c], inplace=True)
    dfc.drop_duplicates(inplace=True)

    # basic imputation
    num_cols = dfc.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = dfc.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in num_cols:
        if dfc[c].isnull().any():
            dfc[c].fillna(dfc[c].median(), inplace=True)
    for c in cat_cols:
        if dfc[c].isnull().any():
            try:
                dfc[c].fillna(dfc[c].mode().iloc[0], inplace=True)
            except Exception:
                dfc[c].fillna("", inplace=True)

    # impute target if exists
    if target_col and target_col in dfc.columns and dfc[target_col].isnull().any():
        if np.issubdtype(dfc[target_col].dtype, np.number):
            dfc[target_col].fillna(dfc[target_col].median(), inplace=True)
        else:
            try:
                dfc[target_col].fillna(dfc[target_col].mode().iloc[0], inplace=True)
            except Exception:
                dfc[target_col].fillna("", inplace=True)

    cleaned_full = dfc.copy()

    # features => one-hot + scale numeric
    if target_col and target_col in cleaned_full.columns:
        feats = cleaned_full.drop(columns=[target_col]).copy()
    else:
        feats = cleaned_full.copy()

    feat_cat = feats.select_dtypes(include=["object", "category"]).columns.tolist()
    feat_num = feats.select_dtypes(include=["number"]).columns.tolist()

    if feat_cat:
        feats = pd.get_dummies(feats, columns=feat_cat, drop_first=True)
    if feat_num:
        scaler = StandardScaler()
        present = [c for c in feat_num if c in feats.columns]
        if present:
            feats[present] = scaler.fit_transform(feats[present])

    cleaned_features = feats
    return cleaned_full, cleaned_features


# ---------- evaluation & training (FAST) ----------
def evaluate_candidates(X: pd.DataFrame, y: pd.Series, task_type: str, reduced_estimators: int = 30):
    results = []
    if task_type == "classification":
        candidates = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForest", RandomForestClassifier(n_estimators=reduced_estimators, n_jobs=-1)),
            ("GBM", GradientBoostingClassifier(n_estimators=reduced_estimators)),
        ]
        scoring = "accuracy"
    else:
        candidates = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestReg", RandomForestRegressor(n_estimators=reduced_estimators, n_jobs=-1)),
            ("GBMReg", GradientBoostingRegressor(n_estimators=reduced_estimators)),
        ]
        scoring = "r2"

    n = X.shape[0]
    cv = min(3, n) if n >= 2 else None

    for name, model in candidates:
        sc = float("-inf")
        try:
            if cv and cv >= 2:
                if task_type == "classification":
                    vc = y.value_counts()
                    if int(vc.min()) >= cv:
                        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                        sc = float(np.mean(scores))
                    else:
                        sc = float("-inf")
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                    sc = float(np.mean(scores))
            else:
                # holdout
                strat = y if (task_type == "classification" and y.nunique() > 1) else None
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)
                model.fit(Xtr, ytr)
                preds = model.predict(Xte)
                sc = float(accuracy_score(yte, preds)) if task_type == "classification" else float(r2_score(yte, preds))
        except Exception:
            sc = float("-inf")
        results.append((name, sc, model))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


# ---------- main exposed function ----------
def process_csv(
    input_csv_path: str,
    outdir: str,
    explicit_target: Optional[str] = None,
    fast_mode: bool = True,
    sample_max_rows: int = 2000,
    reduced_estimators: int = 30,
) -> Dict[str, Any]:
    """
    Process an uploaded CSV:
      - clean and save cleaned_dataset.csv
      - run EDA and generate plots (embedded in report.html)
      - train candidate models on the cleaned dataset (fast defaults)
      - save models in outdir/models/ and best_model.pkl
      - generate report.html in outdir
    Returns a dict training_info with status & metadata.
    """
    safe_mkdir(outdir)
    safe_mkdir(os.path.join(outdir, "models"))
    ret = {"trained": False, "error": None, "saved_models": [], "outdir": outdir}
    try:
        df_raw = pd.read_csv(input_csv_path)
    except Exception as e:
        ret["error"] = f"Failed to read CSV: {e}"
        return ret

    # determine target
    target_col = find_target_column(df_raw, explicit=explicit_target)

    # cleaned artifacts
    cleaned_full, cleaned_features = create_cleaned_full(df_raw, target_col=target_col)
    cleaned_csv_path = os.path.join(outdir, "cleaned_dataset.csv")
    cleaned_full.to_csv(cleaned_csv_path, index=False)

    # EDA calculations (we'll use cleaned_features)
    num_cols = cleaned_features.select_dtypes(include=["number"]).columns.tolist()
    skewed = detect_skewness(cleaned_features, num_cols)
    low_var = detect_low_variance(cleaned_features, num_cols)
    high_corr = detect_high_corr(cleaned_features, num_cols)
    outliers = detect_outliers(cleaned_full, cleaned_full.select_dtypes(include=["number"]).columns.tolist())

    # TRAINING (fast_mode: sample + reduced_estimators)
    training_info = {"trained": False}
    if target_col and target_col in cleaned_full.columns and cleaned_full[target_col].notnull().sum() >= 2:
        # subset rows with non-null target
        df_train = cleaned_full[cleaned_full[target_col].notnull()].copy()
        X = df_train.drop(columns=[target_col]).copy()
        y = df_train[target_col].copy()

        # convert y if numeric-like
        if y.dtype == object:
            try:
                y = pd.to_numeric(y, errors="raise")
            except Exception:
                pass

        # sampling for speed
        if fast_mode and X.shape[0] > sample_max_rows:
            X = X.sample(n=sample_max_rows, random_state=42)
            y = y.loc[X.index]
            # reindex to avoid later alignment issues
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        # transforms: one-hot + scale numeric columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols_X = X.select_dtypes(include=["number"]).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        if num_cols_X:
            scaler = StandardScaler()
            present = [c for c in num_cols_X if c in X.columns]
            if present:
                X[present] = scaler.fit_transform(X[present])

        task_type = infer_task_type(y)
        candidates = evaluate_candidates(X, y, task_type, reduced_estimators=reduced_estimators if fast_mode else reduced_estimators)

        saved = []
        for name, score, model in candidates:
            try:
                model.fit(X, y)
                model_path = os.path.join(outdir, "models", f"{name}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
                saved.append((name, float(score) if score is not None else None, model_path))
            except Exception:
                saved.append((name, None, None))

        if saved:
            saved_sorted = sorted([s for s in saved if s[2] is not None], key=lambda t: (t[1] if t[1] is not None else float("-inf")), reverse=True)
            if saved_sorted:
                best_name, best_score, best_path = saved_sorted[0]
                # copy best to outdir/best_model.pkl
                best_top = os.path.join(outdir, "best_model.pkl")
                try:
                    with open(best_path, "rb") as r, open(best_top, "wb") as w:
                        w.write(r.read())
                except Exception:
                    pass
                try:
                    with open(best_top, "rb") as bf:
                        model_b64 = base64.b64encode(bf.read()).decode("utf-8")
                except Exception:
                    model_b64 = None
                training_info = {
                    "trained": True,
                    "best_name": best_name,
                    "best_score": best_score,
                    "model_b64": model_b64,
                    "task_type": task_type,
                    "target_col": target_col,
                    "metric_name": "accuracy" if task_type == "classification" else "r2",
                    "all_results": [(n, s) for n, s, p in candidates],
                    "saved_models": [(n, p) for n, s, p in saved],
                }
            else:
                training_info = {"trained": False, "error": "No candidate models saved successfully."}
        else:
            training_info = {"trained": False, "error": "No models were attempted or saved."}
    else:
        training_info = {"trained": False, "error": "No target found or not enough non-null target rows to train (need >=2)."}

    # Build report.html (standalone w/ embedded downloads)
    try:
        # Build a simple report (HTML string)
        stats_html = cleaned_features.describe().transpose().to_html(classes="table table-sm", border=0)

        # distributions: chunk numeric columns into figures of up to 8 per fig
        plots_html = ""
        if num_cols:
            max_per_fig = 8
            chunks = [num_cols[i:i + max_per_fig] for i in range(0, len(num_cols), max_per_fig)]
            import math
            for idx, chunk in enumerate(chunks):
                n = len(chunk)
                cols = min(4, n)
                rows = math.ceil(n / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
                axes_flat = axes.flatten() if n > 1 else [axes]
                for i, col in enumerate(chunk):
                    ax = axes_flat[i]
                    try:
                        sns.histplot(cleaned_features[col], kde=True, ax=ax)
                        ax.set_title(col, fontsize=9)
                    except Exception:
                        ax.set_visible(False)
                for j in range(i + 1, len(axes_flat)):
                    axes_flat[j].set_visible(False)
                fig.tight_layout()
                b = fig_to_b64(fig)
                suffix = f" ({idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                plots_html += f"<section class='card'><h3>Distributions{suffix}</h3><img src='data:image/png;base64,{b}'/></section>"

        corr_html = ""
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(min(12, 0.5 * len(num_cols) + 4), min(10, 0.5 * len(num_cols) + 3)))
            sns.heatmap(cleaned_features[num_cols].corr(), cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            corr_b = fig_to_b64(fig)
            corr_html = f"<section class='card'><h3>Correlation Heatmap</h3><img src='data:image/png;base64,{corr_b}'/></section>"

        # embed cleaned csv
        cleaned_csv_b64 = base64.b64encode(open(cleaned_csv_path, "rb").read()).decode("utf-8")
        cleaned_dl = f'<a href="data:text/csv;base64,{cleaned_csv_b64}" download="cleaned_dataset.csv">📥 Download cleaned_dataset.csv</a>'

        # training block html
        train_html = ""
        if training_info.get("trained"):
            rows = ""
            for nm, sc in training_info.get("all_results", []):
                rows += f"<tr><td>{nm}</td><td>{sc if sc is not None else 'FAILED'}</td></tr>"
            saved_list_html = ""
            for nm, p in training_info.get("saved_models", []):
                if p:
                    try:
                        b = base64.b64encode(open(p, "rb").read()).decode("utf-8")
                        saved_list_html += f'<li>{nm} — <a href="data:application/octet-stream;base64,{b}" download="{nm}.pkl">Download</a></li>'
                    except Exception:
                        saved_list_html += f"<li>{nm} — (embed failed)</li>"
                else:
                    saved_list_html += f"<li>{nm} — (failed)</li>"
            best_block = ""
            if training_info.get("model_b64"):
                best_block = f'<p><a href="data:application/octet-stream;base64,{training_info["model_b64"]}" download="best_model.pkl">💾 Download best_model.pkl</a></p>'
            train_html = f"""
            <div class="card">
                <h2>Auto Model Training</h2>
                <p>Task: {training_info.get('task_type')} • Target: {training_info.get('target_col')}</p>
                <p><strong>Selected model:</strong> {training_info.get('best_name')} • <strong>{training_info.get('metric_name')}</strong>: {training_info.get('best_score')}</p>
                {best_block}
                <h3>All tried models</h3>
                <div class="table-wrapper"><table><thead><tr><th>Model</th><th>Score</th></tr></thead><tbody>{rows}</tbody></table></div>
                <h3>Saved model pickles</h3><ul>{saved_list_html}</ul>
            </div>
            """
        else:
            train_html = f'<div class="card"><h2>Auto Model Training</h2><pre>{training_info.get("error")}</pre></div>'

        html = f"""
        <html>
        <head>
            <meta charset="utf-8"/>
            <title>BitStack Report</title>
            <style>
                body {{ font-family: Inter, Roboto, Arial, sans-serif; background: #071028; color: #e6eef8; padding: 18px; }}
                .container {{ max-width: 1100px; margin: 0 auto; }}
                .card {{ background: rgba(255,255,255,0.02); padding: 12px; border-radius: 10px; margin-bottom: 12px; }}
                img {{ max-width: 100%; border-radius: 8px; display:block; margin-top:8px; }}
                table {{ width:100%; border-collapse: collapse; color:#e6eef8; }}
                th, td {{ padding:6px 8px; border:1px solid rgba(255,255,255,0.04); text-align:left; }}
                th {{ background:#0b1a2b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>BitStack Report</h1>
                <div class="card"><h2>Dataset Overview</h2><p>Original rows: {df_raw.shape[0]} • Columns: {df_raw.shape[1]}</p>{cleaned_dl}</div>
                <div class="card"><h2>Key Insights</h2>
                    <ul>
                        <li>Skewed features: {', '.join(skewed.index) if not skewed.empty else 'None'}</li>
                        <li>Low variance features: {', '.join(low_var.index) if not low_var.empty else 'None'}</li>
                        <li>Strong correlation pairs: {len(high_corr)}</li>
                        <li>Outliers (IQR): total across numeric cols: {sum(outliers.values())}</li>
                    </ul>
                </div>
                <div class="card"><h2>Summary Statistics</h2>{stats_html}</div>
                {train_html}
                {corr_html}
                {plots_html}
                <footer style="margin-top:18px; color:#9fb0c9;">BitStack • Generated {now_str()}</footer>
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(outdir, "report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

    except Exception as e:
        ret["error"] = f"Report generation failed: {e}\n{traceback.format_exc()}"
        return ret

    # success
    ret.update({"trained": training_info.get("trained", False), "training_info": training_info, "report": report_path, "cleaned_csv": cleaned_csv_path})
    return ret

