"""Data Lab: real dataset profiling (types, missingness, stats, histograms, correlations, balance, duplicates)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "n/a", "na", "nan", "none", "null", "unknown", "?", "-"}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def infer_kind(s: pd.Series) -> dict:
    """Infer the semantic kind of a column, flagging numbers stored as text."""
    nonnull = s.dropna()
    if pd.api.types.is_bool_dtype(s):
        return {"kind": "binary", "issue": None}
    if pd.api.types.is_numeric_dtype(s):
        uniq = nonnull.nunique()
        return {"kind": "binary" if uniq <= 2 else ("discrete" if pd.api.types.is_integer_dtype(s) or (nonnull % 1 == 0).all() and uniq < 25 else "numeric"),
                "issue": None}
    as_str = nonnull.astype(str)
    avg_words = as_str.str.split().str.len().mean() if len(as_str) else 0
    if avg_words and avg_words > 3:
        return {"kind": "text", "issue": None}
    coerced = pd.to_numeric(as_str.str.strip(), errors="coerce")
    frac_num = coerced.notna().mean() if len(as_str) else 0
    if frac_num > .6:
        bad = [str(b) for b in as_str[coerced.isna()].unique()[:6]]
        return {"kind": "numeric_as_text", "issue": f"{(1 - frac_num) * 100:.0f}% of values are not numbers, e.g. {bad}"}
    norm = as_str.str.strip().str.lower()
    if norm.nunique() < as_str.nunique():
        groups = {}
        for raw in as_str.unique():
            groups.setdefault(str(raw).strip().lower(), []).append(str(raw))
        inconsistent = {k: v for k, v in groups.items() if len(v) > 1}
        return {"kind": "categorical", "issue": f"Inconsistent spellings: {dict(list(inconsistent.items())[:3])}"}
    return {"kind": "categorical", "issue": None}


def histogram(s: pd.Series, bins: int = 20) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"type": "empty"}
    counts, edges = np.histogram(x, bins=min(bins, max(1, int(x.nunique()))))
    return {"type": "numeric", "counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]}


def category_counts(s: pd.Series, top: int = 15) -> dict:
    vc = s.fillna("∅ missing").astype(str).value_counts()
    return {"type": "categorical", "labels": vc.index[:top].tolist(), "counts": vc.values[:top].tolist(), "others": int(vc.values[top:].sum())}


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    with np.errstate(all="ignore"):
        return _profile(df, target)


def _profile(df: pd.DataFrame, target: str | None = None) -> dict:
    cols = []
    disguised_missing = {}
    for c in df.columns:
        s = df[c]
        kind = infer_kind(s)
        miss = int(s.isna().sum())
        if s.dtype == object:
            dm = int(s.dropna().astype(str).str.strip().str.lower().isin(MISSING_TOKENS).sum())
            if dm:
                disguised_missing[c] = dm
        info = dict(name=c, dtype=str(s.dtype), kind=kind["kind"], issue=kind["issue"], missing=miss,
                    missing_pct=round(miss / len(df) * 100, 2), unique=int(s.nunique()), is_target=(c == target),
                    sample=[_clean(v) for v in s.dropna().unique()[:5].tolist()])
        num = pd.to_numeric(s, errors="coerce") if kind["kind"] in ("numeric", "discrete", "binary", "numeric_as_text") else None
        if num is not None and num.notna().any():
            d = num.describe()
            q1, q3 = d["25%"], d["75%"]
            iqr = q3 - q1
            outliers = int(((num < q1 - 3 * iqr) | (num > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
            info["stats"] = {k: _clean(d[k]) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]}
            info["stats"]["skew"] = _clean(num.skew())
            info["outliers"] = outliers
            info["chart"] = histogram(num)
        elif kind["kind"] == "text":
            lens = s.dropna().astype(str).str.split().str.len()
            info["stats"] = {"avg_words": _clean(lens.mean()), "max_words": _clean(lens.max())}
            info["chart"] = histogram(lens)
        else:
            info["chart"] = category_counts(s)
        cols.append(info)

    numeric = df.select_dtypes(include="number")
    corr = None
    if numeric.shape[1] >= 2:
        cm = numeric.corr().round(3).fillna(0)
        if numeric.shape[1] > 20:  # e.g. pixels: keep it readable
            cm = cm.iloc[:20, :20]
        corr = {"columns": cm.columns.tolist(), "matrix": cm.values.tolist()}

    target_info = None
    warnings = []
    if target and target in df.columns:
        t = df[target]
        if t.nunique() <= 20:
            vc = t.value_counts()
            target_info = {"type": "classes", "labels": [str(x) for x in vc.index], "counts": vc.values.tolist(),
                           "majority_share": round(float(vc.iloc[0] / vc.sum()), 4)}
            if vc.iloc[0] / vc.sum() > .9:
                warnings.append(f"Severe class imbalance: '{vc.index[0]}' is {vc.iloc[0] / vc.sum():.1%} of rows. A model predicting only that class gets {vc.iloc[0] / vc.sum():.1%} accuracy.")
        else:
            target_info = {"type": "numeric", "chart": histogram(t), "stats": {k: _clean(v) for k, v in t.describe().items()}}
        # association of each feature with target (absolute Pearson for numeric features; class purity hint otherwise)
        assoc = []
        tnum = pd.to_numeric(t, errors="coerce") if t.nunique() <= 2 or pd.api.types.is_numeric_dtype(t) else None
        if tnum is not None:
            for c in numeric.columns:
                if c == target:
                    continue
                r = numeric[c].corr(tnum)
                if pd.notna(r):
                    assoc.append({"feature": c, "corr": round(float(r), 3)})
            assoc.sort(key=lambda a: -abs(a["corr"]))
            for a in assoc:
                if abs(a["corr"]) > .85:
                    warnings.append(f"'{a['feature']}' correlates {a['corr']:+.2f} with the target — that's suspiciously strong. Could it be leakage?")
        target_info["associations"] = assoc[:25]

    dups = int(df.duplicated().sum())
    if dups:
        warnings.append(f"{dups} exact duplicate rows found.")
    for c in cols:
        if c["issue"]:
            warnings.append(f"Column '{c['name']}': {c['issue']}")
        if c.get("outliers"):
            warnings.append(f"Column '{c['name']}': {c['outliers']} extreme outliers (beyond 3×IQR).")
    for c, k in disguised_missing.items():
        warnings.append(f"Column '{c}': {k} 'disguised' missing values (like 'N/A', '?', '') that pandas does not count as NaN.")
    total_missing = int(df.isna().sum().sum())

    return dict(rows=int(len(df)), columns=int(df.shape[1]), duplicates=dups, total_missing=total_missing,
                memory_kb=round(df.memory_usage(deep=True).sum() / 1024, 1), column_info=cols, correlations=corr,
                target=target_info, warnings=warnings,
                head=[{k: _clean(v) for k, v in row.items()} for row in df.head(12).to_dict(orient="records")])


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, max_points: int = 600) -> dict:
    sub = df[[c for c in {x, y, color} if c]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)
    return {"x": sub[x].round(4).tolist(), "y": sub[y].round(4).tolist(),
            "c": sub[color].astype(str).tolist() if color else None, "corr": _clean(sub[x].corr(sub[y]))}
