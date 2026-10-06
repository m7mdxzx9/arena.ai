"""ML Workbench: real scikit-learn experiments with preprocessing, metrics, diagnostics and code generation."""
from __future__ import annotations

import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import datasets
from .datalab import MISSING_TOKENS, infer_kind


class PipelineError(Exception):
    """An educational error: something a real pipeline would also choke on."""

    def __init__(self, message: str, lesson: str, fix: str):
        super().__init__(message)
        self.message, self.lesson, self.fix = message, lesson, fix

    def to_dict(self):
        return {"error": self.message, "lesson": self.lesson, "fix": self.fix}


# ---------------------------------------------------------------- model registry
def P(name, kind, default, help, **kw):
    return dict(name=name, kind=kind, default=default, help=help, **kw)


MODELS: dict[str, dict] = {
    "dummy": dict(label="Baseline (majority class)", task="classification", cls=DummyClassifier, fixed=dict(strategy="most_frequent"),
                  params=[], interpretability=5, concept="accuracy",
                  notes="Always predicts the most common class. Every real model must beat this."),
    "logistic_regression": dict(label="Logistic Regression", task="classification", cls=LogisticRegression, fixed=dict(max_iter=2000),
                                params=[P("C", "float", 1.0, "Inverse regularisation strength. Smaller = simpler model.", min=0.001, max=100, log=True),
                                        P("class_weight", "choice", "none", "'balanced' up-weights rare classes.", choices=["none", "balanced"])],
                                interpretability=4, concept="logistic_regression",
                                notes="Linear decision boundary; coefficients are readable."),
    "decision_tree": dict(label="Decision Tree", task="classification", cls=DecisionTreeClassifier, fixed={},
                          params=[P("max_depth", "int_or_none", None, "Maximum depth. None = grow until pure (overfit risk).", min=1, max=30),
                                  P("min_samples_leaf", "int", 1, "Minimum samples in each leaf. Larger = smoother.", min=1, max=100),
                                  P("class_weight", "choice", "none", "'balanced' up-weights rare classes.", choices=["none", "balanced"])],
                          interpretability=5, concept="decision_trees", notes="A flowchart of yes/no questions."),
    "random_forest": dict(label="Random Forest", task="classification", cls=RandomForestClassifier, fixed=dict(n_jobs=1),
                          params=[P("n_estimators", "int", 100, "Number of trees.", min=1, max=400),
                                  P("max_depth", "int_or_none", None, "Max depth per tree.", min=1, max=30),
                                  P("min_samples_leaf", "int", 1, "Min samples per leaf.", min=1, max=100),
                                  P("class_weight", "choice", "none", "'balanced' up-weights rare classes.", choices=["none", "balanced"])],
                          interpretability=2, concept="random_forests", notes="Averaged votes of many randomised trees."),
    "gradient_boosting": dict(label="Gradient Boosting", task="classification", cls=GradientBoostingClassifier, fixed={},
                              params=[P("n_estimators", "int", 100, "Number of sequential trees.", min=1, max=400),
                                      P("learning_rate", "float", 0.1, "Shrinks each tree's contribution.", min=0.001, max=1.0, log=True),
                                      P("max_depth", "int", 3, "Depth of each small tree.", min=1, max=10)],
                              interpretability=2, concept="gradient_boosting", notes="Trees fit sequentially to residual errors."),
    "knn": dict(label="k-Nearest Neighbours", task="classification", cls=KNeighborsClassifier, fixed={},
                params=[P("n_neighbors", "int", 5, "How many neighbours vote.", min=1, max=75)],
                interpretability=3, concept="knn", notes="Votes of the k closest training points. Needs scaling!"),
    "naive_bayes": dict(label="Naive Bayes", task="classification", cls=GaussianNB, fixed={}, params=[], interpretability=3, concept="probability",
                        notes="Probabilistic; assumes features are independent given the class. Great for text."),
    # regression
    "dummy_reg": dict(label="Baseline (predict mean)", task="regression", cls=DummyRegressor, fixed=dict(strategy="mean"), params=[],
                      interpretability=5, concept="r2", notes="Always predicts the training mean. R² ≈ 0 by definition."),
    "linear_regression": dict(label="Linear Regression", task="regression", cls=LinearRegression, fixed={}, params=[],
                              interpretability=5, concept="linear_regression", notes="Straight-line (hyperplane) fit by least squares."),
    "ridge": dict(label="Ridge Regression", task="regression", cls=Ridge, fixed={},
                  params=[P("alpha", "float", 1.0, "Regularisation strength. Larger = smaller weights.", min=0.001, max=1000, log=True)],
                  interpretability=5, concept="linear_regression", notes="Linear regression with an L2 penalty."),
    "decision_tree_reg": dict(label="Decision Tree (regression)", task="regression", cls=DecisionTreeRegressor, fixed={},
                              params=[P("max_depth", "int_or_none", None, "Maximum depth.", min=1, max=30),
                                      P("min_samples_leaf", "int", 1, "Min samples per leaf.", min=1, max=100)],
                              interpretability=5, concept="decision_trees", notes="Piecewise-constant predictions."),
    "random_forest_reg": dict(label="Random Forest (regression)", task="regression", cls=RandomForestRegressor, fixed=dict(n_jobs=1),
                              params=[P("n_estimators", "int", 100, "Number of trees.", min=1, max=400),
                                      P("max_depth", "int_or_none", None, "Max depth.", min=1, max=30)],
                              interpretability=2, concept="random_forests", notes="Averaged regression trees."),
    "gradient_boosting_reg": dict(label="Gradient Boosting (regression)", task="regression", cls=GradientBoostingRegressor, fixed={},
                                  params=[P("n_estimators", "int", 150, "Trees.", min=1, max=400),
                                          P("learning_rate", "float", 0.1, "Shrinkage.", min=0.001, max=1.0, log=True),
                                          P("max_depth", "int", 3, "Depth per tree.", min=1, max=10)],
                                  interpretability=2, concept="gradient_boosting", notes="Sequential residual trees."),
    "knn_reg": dict(label="k-NN (regression)", task="regression", cls=KNeighborsRegressor, fixed={},
                    params=[P("n_neighbors", "int", 5, "Neighbours averaged.", min=1, max=75)], interpretability=3, concept="knn",
                    notes="Average of the k closest training targets."),
    # clustering
    "kmeans": dict(label="k-Means", task="clustering", cls=KMeans, fixed=dict(n_init=10),
                   params=[P("n_clusters", "int", 3, "Number of clusters k.", min=1, max=12)], interpretability=4, concept="clustering",
                   notes="Assign to nearest centroid, move centroids, repeat."),
}

DEFAULT_PREP = dict(impute_numeric="median", impute_categorical="most_frequent", scaling="none", encoding="onehot",
                    text_vectorizer="tfidf", dedupe=False, fix_types=False, normalize_categories=False, clip_outliers=False,
                    invalid_to_missing=False)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] learned on the TRAINING data only."""

    def __init__(self, low=0.01, high=0.99):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


# ---------------------------------------------------------------- cleaning (row-level, stateless)
def clean_frame(df: pd.DataFrame, prep: dict, features: list[str], target: str | None) -> tuple[pd.DataFrame, list[str]]:
    log = []
    df = df.copy()
    if prep.get("dedupe"):
        n0 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Removed {n0 - len(df)} duplicate rows.")
    for c in features:
        s = df[c]
        if s.dtype == object:
            kind = infer_kind(s)["kind"]
            if prep.get("fix_types") and kind == "numeric_as_text":
                raw = s.astype(str).str.strip().str.lower()
                raw = raw.where(~raw.isin(MISSING_TOKENS) & s.notna(), None)
                extracted = raw.str.extract(r"(-?\d+(?:\.\d+)?)")[0]
                df[c] = pd.to_numeric(extracted, errors="coerce")
                log.append(f"'{c}': converted text to numbers ({int(df[c].isna().sum())} values became missing).")
            elif prep.get("normalize_categories") and kind == "categorical":
                norm = s.astype(str).str.strip().str.lower().where(s.notna(), None)
                norm = norm.where(~norm.isin(MISSING_TOKENS), None)
                vc = norm.value_counts()
                mapping = {}
                for v in vc.index:  # map abbreviations to a more frequent full label sharing the prefix
                    for full in vc.index:
                        if full != v and len(v) >= 3 and full.startswith(v) and vc[full] > vc[v]:
                            mapping[v] = full
                            break
                norm = norm.replace(mapping)
                before, after = s.nunique(), norm.nunique()
                df[c] = norm
                log.append(f"'{c}': normalised categories {before} → {after} distinct values" + (f" (merged {mapping})" if mapping else "") + ".")
    if prep.get("invalid_to_missing"):
        for c in features:
            if pd.api.types.is_numeric_dtype(df[c]):
                neg = (df[c] < 0).sum()
                if neg and (df[c].dropna() >= 0).mean() > .95:
                    df.loc[df[c] < 0, c] = np.nan
                    log.append(f"'{c}': {int(neg)} impossible negative values set to missing.")
    if target is not None and df[target].isna().any():
        n0 = len(df)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        log.append(f"Dropped {n0 - len(df)} rows with missing target.")
    return df, log


def column_roles(df: pd.DataFrame, features: list[str]) -> dict[str, list[str]]:
    roles = {"numeric": [], "categorical": [], "text": []}
    for c in features:
        kind = infer_kind(df[c])
        k = kind["kind"]
        if k in ("numeric", "discrete", "binary") and pd.api.types.is_numeric_dtype(df[c]):
            roles["numeric"].append(c)
        elif k == "text":
            roles["text"].append(c)
        elif k == "numeric_as_text":
            raise PipelineError(
                f"Column '{c}' looks numeric but contains text values ({kind['issue']}).",
                "Models need numbers. Text like 'N/A' or '34 yrs' inside a numeric column is a *type* error — a classic data-quality problem.",
                "Enable 'Fix types' in preprocessing (converts to numbers; unparseable values become missing), then impute.")
        else:
            roles["categorical"].append(c)
    if len(roles["text"]) > 1:
        raise PipelineError("Only one free-text feature is supported per experiment.",
                            "Each text column needs its own vectorizer.", "Select a single text column.")
    return roles


def build_preprocessor(roles: dict, prep: dict, model_id: str):
    transformers = []
    if roles["numeric"]:
        steps = []
        if prep["impute_numeric"] != "none":
            steps.append(("impute", SimpleImputer(strategy=prep["impute_numeric"])))
        if prep.get("clip_outliers"):
            steps.append(("clip", QuantileClipper()))
        if prep["scaling"] == "standard":
            steps.append(("scale", StandardScaler()))
        elif prep["scaling"] == "minmax":
            steps.append(("scale", MinMaxScaler()))
        transformers.append(("num", Pipeline(steps) if steps else "passthrough", roles["numeric"]))
    if roles["categorical"]:
        steps = []
        if prep["impute_categorical"] != "none":
            steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        if prep["encoding"] == "ordinal":
            steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        else:
            steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(("cat", Pipeline(steps), roles["categorical"]))
    if roles["text"]:
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2)) if prep.get("text_vectorizer", "tfidf") == "tfidf" else CountVectorizer(min_df=2)
        transformers.append(("text", vec, roles["text"][0]))
    sparse_ok = model_id not in ("naive_bayes", "gradient_boosting", "gradient_boosting_reg")
    return ColumnTransformer(transformers, sparse_threshold=0.3 if sparse_ok else 0.0)


def make_model(model_id: str, params: dict, seed: int, roles: dict):
    spec = MODELS[model_id]
    kwargs = dict(spec["fixed"])
    for p in spec["params"]:
        v = params.get(p["name"], p["default"])
        if p["kind"] == "int":
            v = int(np.clip(int(v), p["min"], p["max"]))
        elif p["kind"] == "int_or_none":
            v = None if v in (None, "", "none", "None", 0) else int(np.clip(int(v), p["min"], p["max"]))
        elif p["kind"] == "float":
            v = float(np.clip(float(v), p["min"], p["max"]))
        elif p["kind"] == "choice":
            v = None if v == "none" else v
        kwargs[p["name"]] = v
    cls = spec["cls"]
    if model_id == "naive_bayes" and roles["text"] and not roles["numeric"]:
        cls = MultinomialNB
    if "random_state" in cls().get_params():
        kwargs["random_state"] = seed
    return cls(**kwargs), kwargs


# ---------------------------------------------------------------- experiment
@dataclass
class Experiment:
    dataset: str
    target: str | None
    features: list[str]
    model: str
    params: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=dict)
    test_size: float = 0.2
    seed: int = 42
    cv_folds: int = 0
    threshold: float = 0.5
    extra_rows: int = 0  # Overfitter boss: 'collect more data'

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


def _positive_label(y: pd.Series):
    labels = sorted(y.unique().tolist(), key=str)
    if set(labels) == {0, 1}:
        return 1
    vc = y.value_counts()
    return vc.index[-1]  # minority class


def _clf_metrics(y_true, y_pred, labels, pos, binary):
    avg = "binary" if binary else "macro"
    kw = dict(pos_label=pos) if binary else {}
    return dict(accuracy=float(accuracy_score(y_true, y_pred)),
                balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred, average=avg, zero_division=0, **kw)),
                recall=float(recall_score(y_true, y_pred, average=avg, zero_division=0, **kw)),
                f1=float(f1_score(y_true, y_pred, average=avg, zero_division=0, **kw)))


def _reg_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return dict(mae=float(mean_absolute_error(y_true, y_pred)), mse=mse, rmse=float(np.sqrt(mse)), r2=float(r2_score(y_true, y_pred)))


def _round(obj, nd=4):
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), nd)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def run_experiment(cfg: Experiment, df: pd.DataFrame | None = None) -> dict:
    t_start = time.perf_counter()
    spec = MODELS.get(cfg.model)
    if spec is None:
        raise PipelineError(f"Unknown model '{cfg.model}'.", "", "Pick a model from the list.")
    task = spec["task"]
    prep = {**DEFAULT_PREP, **(cfg.preprocessing or {})}
    if df is None:
        df = datasets.load(cfg.dataset)
        if cfg.extra_rows and cfg.dataset == "overfit_lab":
            df = pd.concat([df, datasets.overfit_lab(n=int(cfg.extra_rows), seed=999)], ignore_index=True)
    feats = [f for f in cfg.features if f in df.columns and f != cfg.target]
    if not feats:
        raise PipelineError("No features selected.", "A model needs at least one input feature (a clue) to learn from.",
                            "Tick one or more feature columns.")
    if task != "clustering":
        if not cfg.target or cfg.target not in df.columns:
            raise PipelineError("Choose a target column.", "Supervised learning needs a label to learn.", "Select the target.")
    df, clean_log = clean_frame(df, prep, feats, cfg.target if task != "clustering" else None)
    roles = column_roles(df, feats)
    pre = build_preprocessor(roles, prep, cfg.model)
    model, model_kwargs = make_model(cfg.model, cfg.params or {}, cfg.seed, roles)
    pipe = Pipeline([("prep", pre), ("model", model)])

    if task == "clustering":
        return _run_clustering(cfg, df, feats, pipe, prep, model_kwargs, clean_log, t_start)

    X, y = df[feats], df[cfg.target]
    if task == "classification" and y.dtype != object and y.nunique() > 20:
        raise PipelineError(f"Target '{cfg.target}' has {y.nunique()} distinct numeric values.",
                            "Classification predicts categories; a continuous target calls for regression.", "Switch to a regression model.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise PipelineError(f"Target '{cfg.target}' is categorical.", "Regression predicts numbers.", "Choose a classification model.")
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    test_size = float(np.clip(cfg.test_size, 0.1, 0.5))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=cfg.seed, stratify=strat)

    try:
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_ms = (time.perf_counter() - t0) * 1000
    except ValueError as e:
        msg = str(e)
        if "NaN" in msg:
            raise PipelineError(f"scikit-learn refused to train: \"{msg.splitlines()[0][:160]}\"",
                                "This model cannot handle missing values (NaN). (Some models — e.g. decision trees and random forests in recent scikit-learn — can; most cannot.)",
                                "Choose an imputation strategy (median is robust to outliers), or drop the columns with missing values.") from e
        if "could not convert string" in msg:
            raise PipelineError("Text found where numbers were expected.", msg[:200], "Enable 'Fix types'.") from e
        raise PipelineError(f"Training failed: {msg[:300]}", "Read the error carefully — it usually names the column.", "Adjust the configuration.") from e

    t0 = time.perf_counter()
    if task == "classification" and hasattr(pipe, "predict_proba") and y.nunique() == 2:
        proba_te = pipe.predict_proba(X_te)
        pos = _positive_label(y)
        pos_idx = list(pipe.classes_).index(pos)
        neg = [c for c in pipe.classes_ if c != pos][0]
        p_te = proba_te[:, pos_idx]
        pred_te = np.where(p_te >= cfg.threshold, pos, neg)
    else:
        p_te = None
        pred_te = pipe.predict(X_te)
    pred_ms = (time.perf_counter() - t0) * 1000
    # latency benchmark: time a real batch of 1000 rows (resampled from the test set) so fixed overhead doesn't dominate
    X_bench = X_te.sample(1000, replace=True, random_state=0) if len(X_te) else X_te
    t1 = time.perf_counter()
    pipe.predict(X_bench)
    per_1k_ms = (time.perf_counter() - t1) * 1000
    if p_te is not None:
        p_tr = pipe.predict_proba(X_tr)[:, pos_idx]
        pred_tr = np.where(p_tr >= cfg.threshold, pos, neg)
    else:
        pred_tr = pipe.predict(X_tr)

    result: dict[str, Any] = dict(task=task, n_train=len(X_tr), n_test=len(X_te), features_used=feats, roles=roles,
                                  cleaning_log=clean_log, train_ms=train_ms, predict_ms_per_1k=per_1k_ms,
                                  model_kwargs={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in model_kwargs.items()})
    try:
        result["model_size_kb"] = len(pickle.dumps(pipe)) / 1024
    except Exception:
        result["model_size_kb"] = None

    if task == "classification":
        labels = sorted(y.unique().tolist(), key=str)
        binary = len(labels) == 2
        pos = _positive_label(y) if binary else None
        result["metrics"] = {"train": _clf_metrics(y_tr, pred_tr, labels, pos, binary), "test": _clf_metrics(y_te, pred_te, labels, pos, binary)}
        result["primary_metric"] = "f1"
        result["positive_label"] = None if pos is None else str(pos)
        result["averaging"] = "binary" if binary else "macro"
        cm = confusion_matrix(y_te, pred_te, labels=labels)
        result["confusion_matrix"] = {"labels": [str(l) for l in labels], "matrix": cm.tolist()}
        base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr).predict(X_te)
        result["baseline"] = _clf_metrics(y_te, base, labels, pos, binary)
        if p_te is not None:
            result["roc_auc"] = float(roc_auc_score((y_te == pos).astype(int), p_te))
            idx = np.arange(len(y_te))[:1500]
            result["test_probabilities"] = {"y_true": [int(v == pos) for v in np.asarray(y_te)[idx]], "p": [round(float(v), 4) for v in p_te[idx]],
                                            "threshold": cfg.threshold}
        per_class = []
        for i, l in enumerate(labels):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
            per_class.append(dict(label=str(l), support=int(cm[i, :].sum()), precision=float(tp / (tp + fp)) if tp + fp else 0.0,
                                  recall=float(tp / (tp + fn)) if tp + fn else 0.0))
        result["per_class"] = per_class
    else:
        result["metrics"] = {"train": _reg_metrics(y_tr, pred_tr), "test": _reg_metrics(y_te, pred_te)}
        result["primary_metric"] = "rmse"
        base = DummyRegressor().fit(X_tr, y_tr).predict(X_te)
        result["baseline"] = _reg_metrics(y_te, base)
        k = min(300, len(y_te))
        result["pred_vs_actual"] = {"actual": np.asarray(y_te)[:k].tolist(), "predicted": np.asarray(pred_te)[:k].tolist()}

    # sample predictions
    sample = X_te.head(8).copy()
    rows = []
    for i, (idx, row) in enumerate(sample.iterrows()):
        r = {k: (None if pd.isna(v) else (v if not isinstance(v, str) else v[:60])) for k, v in row.items()}
        rows.append(dict(features=r, actual=_round(np.asarray(y_te)[i]), predicted=_round(np.asarray(pred_te)[i]),
                         probability=None if p_te is None else round(float(p_te[i]), 3)))
    result["sample_predictions"] = rows

    result["importances"] = _importances(pipe, X_te, y_te, task, cfg.seed)

    if cfg.cv_folds and cfg.cv_folds >= 2:
        folds = int(min(cfg.cv_folds, 10))
        splitter = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed) if task == "classification" else KFold(folds, shuffle=True, random_state=cfg.seed)
        scoring = ("f1" if task == "classification" and y.nunique() == 2 and set(y.unique()) == {0, 1} else
                   "f1_macro" if task == "classification" else "neg_root_mean_squared_error")
        scores = cross_val_score(clone(pipe), X_tr, y_tr, cv=splitter, scoring=scoring)
        if scoring.startswith("neg_"):
            scores = -scores
        result["cv"] = {"folds": folds, "scoring": scoring.replace("neg_", ""), "scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}

    if len(roles["numeric"]) == 2 and not roles["categorical"] and not roles["text"] and task == "classification":
        result["decision_boundary"] = decision_grid(pipe, df, roles["numeric"], y)

    result["diagnosis"] = diagnose(result, task, spec)
    result["interpretability"] = spec["interpretability"]
    result["code"] = generate_code(cfg, prep, roles, model_kwargs, task)
    result["total_ms"] = (time.perf_counter() - t_start) * 1000
    return _round(result)


def _importances(pipe, X_te, y_te, task, seed):
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["prep"].get_feature_names_out())
        names = [re.sub(r"^(num|cat|text)__", "", n) for n in names]
    except Exception:
        names = None
    vals = None
    method = None
    if hasattr(model, "feature_importances_"):
        vals, method = model.feature_importances_, "Impurity-based importance (trees)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        vals = coef[0] if coef.ndim == 2 and coef.shape[0] == 1 else (np.abs(coef).mean(0) if coef.ndim == 2 else coef)
        method = "Model coefficients (sign = direction; scale features to compare magnitudes fairly)"
    if vals is not None and names is not None and len(names) == len(vals):
        order = np.argsort(-np.abs(vals))[:15]
        return {"method": method, "items": [{"feature": names[i], "value": float(vals[i])} for i in order]}
    if len(X_te) <= 1000 and task != "clustering":
        try:
            pi = permutation_importance(pipe, X_te, y_te, n_repeats=3, random_state=seed)
            order = np.argsort(-pi.importances_mean)[:15]
            return {"method": "Permutation importance (drop in score when a column is shuffled)",
                    "items": [{"feature": X_te.columns[i], "value": float(pi.importances_mean[i])} for i in order]}
        except Exception:
            return None
    return None


def decision_grid(pipe, df, cols, y, res=60):
    x0, x1 = df[cols[0]].astype(float), df[cols[1]].astype(float)
    pad0, pad1 = (x0.max() - x0.min()) * .08, (x1.max() - x1.min()) * .08
    g0 = np.linspace(x0.min() - pad0, x0.max() + pad0, res)
    g1 = np.linspace(x1.min() - pad1, x1.max() + pad1, res)
    G0, G1 = np.meshgrid(g0, g1)
    grid = pd.DataFrame({cols[0]: G0.ravel(), cols[1]: G1.ravel()})
    classes = list(pipe.classes_)
    if hasattr(pipe, "predict_proba") and len(classes) == 2:
        z = pipe.predict_proba(grid)[:, 1]
    else:
        z = np.array([classes.index(c) for c in pipe.predict(grid)], dtype=float)
    sample = df.sample(min(400, len(df)), random_state=0)
    return {"x_range": [float(g0[0]), float(g0[-1])], "y_range": [float(g1[0]), float(g1[-1])], "res": res,
            "z": np.round(z, 3).reshape(res, res).tolist(), "classes": [str(c) for c in classes], "features": cols,
            "points": {"x": sample[cols[0]].tolist(), "y": sample[cols[1]].tolist(), "label": [classes.index(v) for v in y.loc[sample.index]]}}


def _run_clustering(cfg, df, feats, pipe, prep, model_kwargs, clean_log, t_start):
    X = df[feats]
    t0 = time.perf_counter()
    labels = pipe.fit_predict(X)
    train_ms = (time.perf_counter() - t0) * 1000
    Z = pipe.named_steps["prep"].transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    k = model_kwargs.get("n_clusters", 3)
    sil = float(silhouette_score(Z, labels)) if 1 < k < len(X) else None
    proj = PCA(2, random_state=cfg.seed).fit_transform(Z) if Z.shape[1] > 2 else Z
    profile = X.assign(cluster=labels).groupby("cluster").mean(numeric_only=True).round(2)
    # elbow: real inertia for k = 1..8
    elbow = []
    for kk in range(1, 9):
        km = KMeans(kk, n_init=5, random_state=cfg.seed).fit(Z)
        elbow.append({"k": kk, "inertia": float(km.inertia_)})
    res = dict(task="clustering", n_train=len(X), n_test=0, features_used=feats, cleaning_log=clean_log, train_ms=train_ms,
               metrics={"train": {"inertia": float(pipe.named_steps["model"].inertia_), "silhouette": sil}, "test": {}},
               primary_metric="silhouette", cluster_sizes=np.bincount(labels).tolist(),
               projection={"x": proj[:, 0].tolist(), "y": proj[:, 1].tolist(), "cluster": labels.tolist(),
                           "note": "2D PCA projection of the preprocessed features" if Z.shape[1] > 2 else "Raw 2D features"},
               cluster_profiles={"columns": profile.columns.tolist(), "rows": profile.reset_index().values.tolist()}, elbow=elbow,
               model_kwargs={k: v for k, v in model_kwargs.items() if isinstance(v, (int, float, str))}, interpretability=4,
               diagnosis=[{"level": "info", "code": "clustering", "text": "Clustering has no ground-truth labels. Silhouette (−1…1) measures how well-separated clusters are; inertia always falls as k grows — look for the elbow."}]
               + ([] if prep["scaling"] != "none" else [{"level": "warn", "code": "unscaled", "text": "Features are unscaled: k-means uses distances, so large-range features (like income) dominate. Try standard scaling."}]),
               code=generate_code(cfg, prep, {"numeric": feats, "categorical": [], "text": []}, model_kwargs, "clustering"))
    res["total_ms"] = (time.perf_counter() - t_start) * 1000
    return _round(res)


def diagnose(result, task, spec) -> list[dict]:
    out = []
    tr, te = result["metrics"]["train"], result["metrics"]["test"]
    if task == "classification":
        gap = tr["accuracy"] - te["accuracy"]
        if gap > .12:
            out.append({"level": "warn", "code": "overfit", "text": f"Overfitting: train accuracy {tr['accuracy']:.2f} vs test {te['accuracy']:.2f} (gap {gap:.2f}). Try a simpler model (max_depth, min_samples_leaf), fewer noisy features, regularisation, or more data."})
        if te["accuracy"] > .985 and spec["label"] != "Baseline (majority class)" and result["baseline"]["accuracy"] < .9:
            out.append({"level": "danger", "code": "too_good", "text": "Near-perfect test score on a realistic problem. Before celebrating: check for leakage — features that are only known after the outcome, or derived from the target."})
        b = result["baseline"]
        if b["accuracy"] > .85:
            out.append({"level": "warn", "code": "imbalance", "text": f"Imbalanced classes: always predicting the majority gives {b['accuracy']:.1%} accuracy. Judge the model by recall/precision/F1 on the rare class instead."})
        if te["accuracy"] <= b["accuracy"] + .01 and spec["label"] != "Baseline (majority class)":
            out.append({"level": "warn", "code": "no_better", "text": "Barely better than the majority baseline. The features may carry little signal, or the model ignores the minority class."})
        if tr["accuracy"] < .7 and gap < .05 and b["accuracy"] < .7:
            out.append({"level": "info", "code": "underfit", "text": "Train and test scores are both low and close: possible underfitting. Try a more flexible model or better features."})
    else:
        if tr["r2"] - te["r2"] > .15:
            out.append({"level": "warn", "code": "overfit", "text": f"Overfitting: train R² {tr['r2']:.2f} vs test R² {te['r2']:.2f}."})
        if te["r2"] < 0:
            out.append({"level": "danger", "code": "worse_than_mean", "text": "Negative test R²: worse than always predicting the mean."})
        if te["rmse"] > 1.5 * te["mae"]:
            out.append({"level": "info", "code": "big_errors", "text": "RMSE is much larger than MAE: a few predictions are very wrong (large errors)."})
    if not out:
        out.append({"level": "ok", "code": "healthy", "text": "No obvious red flags. Compare against other models and check per-class metrics."})
    return out


# ---------------------------------------------------------------- code generation
def generate_code(cfg: Experiment, prep: dict, roles: dict, model_kwargs: dict, task: str) -> str:
    spec = MODELS[cfg.model]
    cls = spec["cls"]
    if cfg.model == "naive_bayes" and roles.get("text") and not roles.get("numeric"):
        cls = MultinomialNB
    mod = cls.__module__.split("._")[0]
    kw = ",\n    ".join(f"{k}={v!r}" for k, v in model_kwargs.items())
    lines = ["import pandas as pd", "from sklearn.pipeline import Pipeline", "from sklearn.compose import ColumnTransformer"]
    imports = {f"from {mod} import {cls.__name__}"}
    if task != "clustering":
        imports.add("from sklearn.model_selection import train_test_split")
    num_steps, cat_steps = [], []
    if roles.get("numeric"):
        if prep["impute_numeric"] != "none":
            imports.add("from sklearn.impute import SimpleImputer")
            num_steps.append(f"('impute', SimpleImputer(strategy={prep['impute_numeric']!r}))")
        if prep["scaling"] == "standard":
            imports.add("from sklearn.preprocessing import StandardScaler"); num_steps.append("('scale', StandardScaler())")
        elif prep["scaling"] == "minmax":
            imports.add("from sklearn.preprocessing import MinMaxScaler"); num_steps.append("('scale', MinMaxScaler())")
    if roles.get("categorical"):
        imports.add("from sklearn.impute import SimpleImputer")
        cat_steps.append("('impute', SimpleImputer(strategy='most_frequent'))")
        if prep["encoding"] == "ordinal":
            imports.add("from sklearn.preprocessing import OrdinalEncoder")
            cat_steps.append("('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))")
        else:
            imports.add("from sklearn.preprocessing import OneHotEncoder")
            cat_steps.append("('encode', OneHotEncoder(handle_unknown='ignore'))")
    if roles.get("text"):
        imports.add("from sklearn.feature_extraction.text import TfidfVectorizer" if prep.get("text_vectorizer", "tfidf") == "tfidf" else "from sklearn.feature_extraction.text import CountVectorizer")
    lines += sorted(imports)
    lines += ["", f"df = pd.read_csv('{cfg.dataset}.csv')"]
    if prep.get("dedupe"):
        lines.append("df = df.drop_duplicates()  # remove duplicate rows BEFORE splitting")
    if prep.get("fix_types"):
        lines.append("# Fix types: extract numbers from text like '34 yrs'; 'N/A' -> NaN")
        lines.append("for col in df.select_dtypes('object'):\n    nums = pd.to_numeric(df[col].astype(str).str.extract(r'(-?\\d+\\.?\\d*)')[0], errors='coerce')\n    if nums.notna().mean() > 0.6:\n        df[col] = nums")
    if prep.get("normalize_categories"):
        lines.append("for col in df.select_dtypes('object'):\n    df[col] = df[col].str.strip().str.lower()  # 'Basic ' -> 'basic'")
    lines.append(f"features = {cfg.features!r}")
    if task != "clustering":
        lines.append(f"X, y = df[features], df[{cfg.target!r}]")
    else:
        lines.append("X = df[features]")
    parts = []
    if roles.get("numeric"):
        parts.append(f"    ('num', Pipeline([{', '.join(num_steps)}]) , {roles['numeric']!r})" if num_steps else f"    ('num', 'passthrough', {roles['numeric']!r})")
    if roles.get("categorical"):
        parts.append(f"    ('cat', Pipeline([{', '.join(cat_steps)}]), {roles['categorical']!r})")
    if roles.get("text"):
        vec = "TfidfVectorizer(min_df=2, ngram_range=(1, 2))" if prep.get("text_vectorizer", "tfidf") == "tfidf" else "CountVectorizer(min_df=2)"
        parts.append(f"    ('text', {vec}, {roles['text'][0]!r})")
    lines += ["", "preprocess = ColumnTransformer([", ",\n".join(parts), "])", "",
              f"model = Pipeline([\n    ('prep', preprocess),\n    ('model', {cls.__name__}(\n    {kw}\n    )),\n])"]
    if task == "clustering":
        lines += ["", "clusters = model.fit_predict(X)", "print(pd.Series(clusters).value_counts())"]
        return "\n".join(lines)
    strat = ", stratify=y" if task == "classification" else ""
    lines += ["", f"X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size={cfg.test_size}, random_state={cfg.seed}{strat})",
              "model.fit(X_train, y_train)  # preprocessing is fit on TRAIN only", "y_pred = model.predict(X_test)", ""]
    if task == "classification":
        lines += ["from sklearn.metrics import classification_report, confusion_matrix", "print(confusion_matrix(y_test, y_pred))", "print(classification_report(y_test, y_pred))"]
    else:
        lines += ["from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score",
                  "print('MAE ', mean_absolute_error(y_test, y_pred))", "print('RMSE', root_mean_squared_error(y_test, y_pred))", "print('R2  ', r2_score(y_test, y_pred))"]
    return "\n".join(lines)


def model_catalog():
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("cls",)} for k, v in MODELS.items()}


def compare_runs(runs: list[dict]) -> dict:
    """Trade-off analysis across runs — not just 'bigger number wins'."""
    if not runs:
        return {"rows": [], "insights": []}
    task = runs[0]["result"]["task"]
    rows, insights = [], []
    for r in runs:
        res = r["result"]
        m = res["metrics"]["test"]; tr = res["metrics"]["train"]
        primary = m.get("f1") if task == "classification" else m.get("rmse") if task == "regression" else res["metrics"]["train"].get("silhouette")
        gap = (tr.get("accuracy", 0) - m.get("accuracy", 0)) if task == "classification" else (tr.get("r2", 0) - m.get("r2", 0)) if task == "regression" else 0
        rows.append(dict(id=r["id"], name=r.get("name") or f"Run #{r['id']}", model=MODELS[r["config"]["model"]]["label"], primary=primary,
                         metrics=m, gap=gap, train_ms=res.get("train_ms"), predict_ms_per_1k=res.get("predict_ms_per_1k"),
                         model_size_kb=res.get("model_size_kb"), interpretability=res.get("interpretability"), cv=res.get("cv"),
                         n_features=len(res.get("features_used", []))))
    if task in ("classification", "regression"):
        better = max if task == "classification" else min
        best = better(rows, key=lambda x: x["primary"] if x["primary"] is not None else (-1e18 if task == "classification" else 1e18))
        insights.append(f"Highest test score: {best['name']} ({best['model']}).")
        for x in rows:
            if x["gap"] > .1:
                insights.append(f"{x['name']} shows a large train–test gap ({x['gap']:.2f}) → its score may not generalise as well as it looks.")
        cvs = [x for x in rows if x["cv"]]
        if len(cvs) >= 2:
            a, b = sorted(cvs, key=lambda x: -x["cv"]["mean"])[:2]
            if abs(a["cv"]["mean"] - b["cv"]["mean"]) < max(a["cv"]["std"], b["cv"]["std"]):
                insights.append(f"Cross-validation: {a['name']} and {b['name']} differ by less than one standard deviation — the difference may be noise.")
        fastest = min(rows, key=lambda x: x["predict_ms_per_1k"] or 1e9)
        if fastest["id"] != best["id"]:
            ratio = (best["predict_ms_per_1k"] or 0) / max(fastest["predict_ms_per_1k"] or 1e-9, 1e-9)
            if ratio > 1.5:
                insights.append(f"{fastest['name']} predicts ~{ratio:.0f}× faster than {best['name']}. If latency matters, the small score difference may not be worth it.")
        interp = max(rows, key=lambda x: x["interpretability"] or 0)
        if interp["id"] != best["id"] and (interp["interpretability"] or 0) > (best["interpretability"] or 0):
            insights.append(f"{interp['name']} ({interp['model']}) is more interpretable — useful when decisions must be explained (e.g., to students or regulators).")
        small = min(rows, key=lambda x: x["model_size_kb"] or 1e9)
        if small["id"] != best["id"] and (best["model_size_kb"] or 0) > 5 * (small["model_size_kb"] or 1):
            insights.append(f"{best['name']} is {best['model_size_kb'] / max(small['model_size_kb'], 1e-6):.0f}× larger in memory than {small['name']}.")
        datasets_used = {r["config"]["dataset"] for r in runs}
        seeds = {r["config"].get("seed") for r in runs}
        splits = {r["config"].get("test_size") for r in runs}
        if len(datasets_used) > 1:
            insights.append("⚠ These runs use different datasets — their metrics are not directly comparable.")
        if len(seeds) > 1 or len(splits) > 1:
            insights.append("⚠ Different random seeds or split sizes: part of the score difference comes from a different test set. Fix the seed for fair comparisons.")
    return {"task": task, "rows": _round(rows), "insights": insights}
