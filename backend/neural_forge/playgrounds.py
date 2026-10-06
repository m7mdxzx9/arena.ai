"""Computation behind the interactive visual widgets. Everything here is real computation on real data."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import datasets, rag
from .corpus import DOCS


# ------------------------------------------------------------- BPE tokenizer
def bpe_train(num_merges: int = 60, corpus: str | None = None) -> dict:
    text = corpus or " ".join(d["text"] for d in DOCS)
    words = Counter(re.findall(r"[a-zA-Z]+", text.lower()))
    vocab = {tuple(w) + ("</w>",): c for w, c in words.items()}
    merges = []
    for _ in range(int(np.clip(num_merges, 0, 400))):
        pairs = Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        merges.append({"pair": [a, b], "freq": freq})
        new_vocab = {}
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(sym[i]); i += 1
            new_vocab[tuple(out)] = c
        vocab = new_vocab
    return {"merges": merges}


def bpe_tokenize(text: str, merges: list[dict]) -> list[str]:
    toks = []
    for w in re.findall(r"[a-zA-Z]+|[0-9]|[^\sa-zA-Z0-9]", text.lower()):
        if not w.isalpha():
            toks.append(w); continue
        sym = list(w) + ["</w>"]
        for m in merges:
            a, b = m["pair"]
            i, out = 0, []
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(sym[i]); i += 1
            sym = out
        toks.extend(sym)
    return toks


def tokenizer_demo(text: str, num_merges: int = 60) -> dict:
    model = bpe_train(num_merges)
    toks = bpe_tokenize(text, model["merges"])
    vocab = sorted(set(toks))
    return {"text": text, "char_tokens": len([c for c in text if not c.isspace()]), "word_tokens": len(text.split()),
            "bpe_tokens": toks, "ids": [vocab.index(t) for t in toks], "num_merges": len(model["merges"]),
            "first_merges": model["merges"][:25]}


# ------------------------------------------------------------- attention
def attention_demo(sentence: str) -> dict:
    """One attention head with identity projections over co-occurrence word vectors (LSA term vectors).

    Shows the *mechanics* (dot products → scaled → softmax → weighted mix). A trained transformer learns
    W_Q, W_K, W_V and would attend differently."""
    idx = rag.get_index(60, 10, "lsa")
    vocab = idx.vec.vocabulary_
    term_vecs = idx.svd.components_.T  # (n_terms, k)
    words = re.findall(r"[A-Za-z0-9\-]+", sentence)[:14]
    vecs, known = [], []
    rng = np.random.default_rng(0)
    for w in words:
        j = vocab.get(w.lower())
        if j is not None:
            vecs.append(term_vecs[j]); known.append(True)
        else:  # unknown/stop word: small random vector (it has no learned meaning in this corpus)
            vecs.append(rng.normal(0, 0.02, term_vecs.shape[1])); known.append(False)
    E = np.array(vecs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    d = E.shape[1]
    scores = (E @ E.T) / np.sqrt(d) * 8  # temperature so the softmax is visibly peaked
    W = np.exp(scores - scores.max(1, keepdims=True))
    W /= W.sum(1, keepdims=True)
    return {"tokens": words, "known": known, "weights": np.round(W, 3).tolist(),
            "note": "Single head, identity Q/K/V projections over corpus co-occurrence vectors. Illustrates how attention weights are computed, not what a trained LLM attends to."}


# ------------------------------------------------------------- k-means step by step
def kmeans_steps(dataset: str = "blobs", k: int = 3, seed: int = 0, max_iter: int = 10) -> dict:
    if dataset == "customer_segments":
        df = datasets.load("customer_segments")
        X = df[["annual_income_k", "spending_score"]].values.astype(float)
        names = ["annual_income_k", "spending_score"]
    else:
        df = datasets.load(dataset)
        X = df[["x1", "x2"]].values.astype(float)
        names = ["x1", "x2"]
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), int(np.clip(k, 1, 8)), replace=False)].copy()
    steps = []
    for it in range(max_iter):
        dists = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        assign = dists.argmin(1)
        inertia = float(dists[np.arange(len(X)), assign].sum())
        steps.append({"iter": it, "centroids": C.round(4).tolist(), "assign": assign.tolist(), "inertia": round(inertia, 3)})
        newC = np.array([X[assign == j].mean(0) if (assign == j).any() else C[j] for j in range(len(C))])
        if np.allclose(newC, C):
            break
        C = newC
    sk = KMeans(len(C), init=np.array(steps[0]["centroids"]), n_init=1, max_iter=max_iter).fit(X)
    return {"points": X.round(4).tolist(), "features": names, "steps": steps, "converged": len(steps) < max_iter,
            "sklearn_inertia_check": round(float(sk.inertia_), 3)}


# ------------------------------------------------------------- linear regression playground
def linreg_points(seed: int = 3, n: int = 30) -> dict:
    df = datasets.load("house_prices").sample(n, random_state=seed)
    x = df["size_sqm"].values.astype(float)
    y = (df["price"].values / 1000).astype(float)
    m = LinearRegression().fit(x[:, None], y)
    pred = m.predict(x[:, None])
    return {"x": x.tolist(), "y": y.round(2).tolist(), "x_label": "Size (m²)", "y_label": "Price (k€)",
            "fit": {"slope": round(float(m.coef_[0]), 4), "intercept": round(float(m.intercept_), 3),
                    "mse": round(float(np.mean((y - pred) ** 2)), 3)}}


# ------------------------------------------------------------- train/test split & CV
def split_demo(test_size: float = 0.2, seed: int = 42, stratify: bool = True) -> dict:
    df = datasets.load("moons")
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=float(np.clip(test_size, .05, .9)), random_state=seed,
                              stratify=df["label"] if stratify else None)
    split = np.zeros(len(df), int); split[te] = 1
    return {"x": df["x1"].tolist(), "y": df["x2"].tolist(), "label": df["label"].tolist(), "split": split.tolist(),
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_class_share": round(float(df["label"].values[te].mean()), 3), "train_class_share": round(float(df["label"].values[tr].mean()), 3)}


def cv_demo(folds: int = 5, seed: int = 0) -> dict:
    df = datasets.load("moons")
    X, y = df[["x1", "x2"]].values, df["label"].values
    out = []
    for i, (tr, va) in enumerate(KFold(int(np.clip(folds, 2, 10)), shuffle=True, random_state=seed).split(X)):
        m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[tr], y[tr])
        out.append({"fold": i + 1, "val_idx": va.tolist(), "score": round(float(m.score(X[va], y[va])), 4)})
    scores = [f["score"] for f in out]
    return {"n": len(X), "folds": out, "mean": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)}


# ------------------------------------------------------------- overfitting curve (real sweep)
def overfit_curve(dataset: str = "student_success") -> dict:
    df = datasets.load(dataset)
    if dataset == "student_success":
        feats, target = ["study_hours", "attendance", "previous_grade", "sleep_hours"], "passed"
    else:
        target = datasets.REGISTRY[dataset]["target"]
        feats = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].fillna(df[feats].median()).values
    y = df[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    rows = []
    for d in list(range(1, 16)) + [None]:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
        rows.append({"depth": d if d else "∞", "train": round(float(m.score(Xtr, ytr)), 4), "test": round(float(m.score(Xte, yte)), 4),
                     "leaves": int(m.get_n_leaves())})
    best = max(rows, key=lambda r: r["test"])
    return {"rows": rows, "best_depth": best["depth"], "dataset": dataset}


# ------------------------------------------------------------- threshold explorer
def threshold_data(dataset: str = "fraud") -> dict:
    df = datasets.load(dataset)
    target = datasets.REGISTRY[dataset]["target"]
    feats = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].fillna(df[feats].median()).values
    y = df[target].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return {"y_true": yte.astype(int).tolist(), "p": np.round(p, 4).tolist(), "dataset": dataset, "positive_share": round(float(y.mean()), 4),
            "model": "LogisticRegression (standardised features), 70/30 stratified split"}


# ------------------------------------------------------------- convolution
KERNELS = {
    "vertical_edge": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "horizontal_edge": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "blur": [[1 / 9] * 3] * 3,
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "identity": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
}


def conv_demo(index: int = 0, kernel: str | list = "vertical_edge") -> dict:
    df = datasets.load("digits")
    row = df.iloc[int(index) % len(df)]
    img = row[[f"px_{i}" for i in range(64)]].values.astype(float).reshape(8, 8)
    K = np.array(KERNELS.get(kernel, KERNELS["vertical_edge"]) if isinstance(kernel, str) else kernel, float)
    pad = np.pad(img, 1)
    out = np.zeros_like(img)
    for i in range(8):
        for j in range(8):
            out[i, j] = float((pad[i:i + 3, j:j + 3] * K).sum())
    relu = np.maximum(out, 0)
    pooled = relu.reshape(4, 2, 4, 2).max(axis=(1, 3))
    return {"label": int(row["digit"]), "image": img.tolist(), "kernel": K.round(3).tolist(), "feature_map": out.round(2).tolist(),
            "relu": relu.round(2).tolist(), "pooled": pooled.round(2).tolist(), "kernels": list(KERNELS)}


# ------------------------------------------------------------- language model
def lm_demo(word: str = "the", temperature: float = 1.0, seed: int = 0) -> dict:
    model = rag.lm()
    rng = np.random.default_rng(seed)
    samples = [model.generate([word.lower()], temperature=temperature, rng=rng, max_len=16) for _ in range(3)]
    return {"word": word, "next": model.distribution(word, temperature), "samples": samples, "temperature": temperature,
            "note": "A real bigram model trained on the 14-document campus handbook: it only knows which word tends to follow which."}


def distribution_samples(kind: str = "normal", n: int = 2000, seed: int = 0) -> dict:
    r = np.random.default_rng(seed)
    x = {"normal": r.normal(50, 10, n), "right_skewed": r.lognormal(3, .6, n), "uniform": r.uniform(0, 100, n),
         "bimodal": np.r_[r.normal(30, 6, n // 2), r.normal(70, 6, n - n // 2)]}.get(kind)
    if x is None:
        raise ValueError(kind)
    counts, edges = np.histogram(x, bins=30)
    return {"kind": kind, "counts": counts.tolist(), "edges": edges.round(3).tolist(), "mean": round(float(x.mean()), 3),
            "median": round(float(np.median(x)), 3), "std": round(float(x.std()), 3)}


def onehot_demo() -> dict:
    df = datasets.load("customer_churn")[["contract", "internet"]].head(6)
    enc = pd.get_dummies(df, dtype=int)
    return {"before": df.to_dict(orient="records"), "after_columns": enc.columns.tolist(), "after": enc.values.tolist()}


def scaling_demo() -> dict:
    df = datasets.load("medical")[["age", "cholesterol"]].head(200)
    s = (df - df.mean()) / df.std()
    return {"raw": {"x": df["age"].tolist(), "y": df["cholesterol"].tolist()}, "scaled": {"x": s["age"].round(3).tolist(), "y": s["cholesterol"].round(3).tolist()},
            "raw_stats": df.describe().loc[["mean", "std"]].round(2).to_dict(), "names": ["age", "cholesterol"]}
