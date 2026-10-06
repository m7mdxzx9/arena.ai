"""Neural Network Lab: a transparent, real multilayer perceptron written in NumPy.

Every number shown in the lab comes from this code: forward pass, softmax cross-entropy loss,
backpropagation, SGD / Momentum / Adam updates, inverted dropout and L2 weight decay.
The implementation is small on purpose so learners can read it (see /api/nn/source).
A gradient check in the test-suite verifies backprop against numerical derivatives.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import datasets

ACTS = {
    "relu": (lambda z: np.maximum(0, z), lambda z, a: (z > 0).astype(z.dtype)),
    "tanh": (np.tanh, lambda z, a: 1 - a ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60))), lambda z, a: a * (1 - a)),
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
}

NN_DATASETS = {
    "moons": ("moons", ["x1", "x2"], "label"), "circles": ("circles", ["x1", "x2"], "label"),
    "spiral": ("spiral", ["x1", "x2"], "label"), "xor": ("xor", ["x1", "x2"], "label"),
    "linear2d": ("linear2d", ["x1", "x2"], "label"), "blobs": ("blobs", ["x1", "x2"], "label"),
    "student_success": ("student_success", ["study_hours", "attendance", "previous_grade", "sleep_hours"], "passed"),
    "digits": ("digits", [f"px_{i}" for i in range(64)], "digit"),
    "overfit_lab": ("overfit_lab", None, "label"),
}

MAX_WORK = 4e8  # cap on (samples × epochs × params) to keep runs fast and the server responsive


class MLP:
    def __init__(self, sizes, activation="relu", seed=0, dropout=0.0, l2=0.0):
        self.rng = np.random.default_rng(seed)
        self.act, self.dact = ACTS[activation]
        self.activation, self.dropout, self.l2 = activation, float(dropout), float(l2)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2 / fan_in) if activation == "relu" else np.sqrt(1 / fan_in)  # He / Xavier-style init
            self.W.append(self.rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def forward(self, X, train=False):
        cache = {"a": [X], "z": [], "masks": []}
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            cache["z"].append(z)
            if i < len(self.W) - 1:
                a = self.act(z)
                if train and self.dropout > 0:
                    mask = (self.rng.random(a.shape) >= self.dropout) / (1 - self.dropout)  # inverted dropout
                    a = a * mask
                    cache["masks"].append(mask)
                else:
                    cache["masks"].append(None)
            else:
                z = z - z.max(1, keepdims=True)
                e = np.exp(z)
                a = e / e.sum(1, keepdims=True)  # softmax
            cache["a"].append(a)
        return a, cache

    def loss(self, P, Y):
        ce = -np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1)), axis=1))
        return ce + 0.5 * self.l2 * sum(np.sum(W ** 2) for W in self.W)

    def backward(self, cache, Y):
        n = Y.shape[0]
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        delta = (cache["a"][-1] - Y) / n  # d(CE)/dz for softmax + cross-entropy
        for i in reversed(range(len(self.W))):
            grads_W[i] = cache["a"][i].T @ delta + self.l2 * self.W[i]
            grads_b[i] = delta.sum(0)
            if i > 0:
                da = delta @ self.W[i].T
                if cache["masks"][i - 1] is not None:
                    da = da * cache["masks"][i - 1]
                    a_prev = cache["a"][i] / np.where(cache["masks"][i - 1] == 0, 1, cache["masks"][i - 1])
                else:
                    a_prev = cache["a"][i]
                delta = da * self.dact(cache["z"][i - 1], a_prev)
        return grads_W, grads_b

    def predict_proba(self, X):
        return self.forward(X)[0]


class Optimizer:
    def __init__(self, kind, lr, params_shapes):
        self.kind, self.lr, self.t = kind, lr, 0
        self.m = [np.zeros(s) for s in params_shapes]
        self.v = [np.zeros(s) for s in params_shapes]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.kind == "sgd":
                p -= self.lr * g
            elif self.kind == "momentum":
                self.m[i] = 0.9 * self.m[i] + g
                p -= self.lr * self.m[i]
            else:  # adam
                self.m[i] = 0.9 * self.m[i] + 0.1 * g
                self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                mh = self.m[i] / (1 - 0.9 ** self.t)
                vh = self.v[i] / (1 - 0.999 ** self.t)
                p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def load_nn_data(name: str):
    ds_id, feats, target = NN_DATASETS[name]
    df = datasets.load(ds_id)
    if feats is None:
        feats = [c for c in df.columns if c != target]
    X = df[feats].astype(float)
    X = X.fillna(X.median()).values
    y_raw = df[target].values
    classes = sorted(np.unique(y_raw).tolist())
    y = np.array([classes.index(v) for v in y_raw])
    return X, y, feats, [str(c) for c in classes]


def train(config: dict) -> dict:
    t_start = time.perf_counter()
    name = config.get("dataset", "moons")
    hidden = [int(np.clip(h, 1, 128)) for h in config.get("hidden", [8, 8])][:5]
    act = config.get("activation", "relu")
    if act not in ACTS:
        raise ValueError(f"Unknown activation {act}")
    lr = float(np.clip(config.get("lr", 0.01), 1e-6, 50))
    opt_kind = config.get("optimizer", "adam")
    batch = int(np.clip(config.get("batch_size", 32), 1, 4096))
    epochs = int(np.clip(config.get("epochs", 100), 1, 1000))
    dropout = float(np.clip(config.get("dropout", 0.0), 0, 0.95))
    l2 = float(np.clip(config.get("l2", 0.0), 0, 1))
    seed = int(config.get("seed", 42))
    scale = bool(config.get("scale_inputs", True))
    val_size = float(np.clip(config.get("val_size", 0.3), 0.1, 0.5))
    train_limit = config.get("train_limit")

    X, y, feats, classes = load_nn_data(name)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=val_size, random_state=seed, stratify=y)
    if train_limit:
        X_tr, y_tr = X_tr[: int(train_limit)], y_tr[: int(train_limit)]
    scaler = StandardScaler().fit(X_tr) if scale else None
    if scaler is not None:
        X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)
    else:
        X_tr_s, X_va_s = X_tr, X_va
    k = len(classes)
    Y_tr, Y_va = np.eye(k)[y_tr], np.eye(k)[y_va]
    sizes = [X.shape[1]] + hidden + [k]
    net = MLP(sizes, act, seed=seed, dropout=dropout, l2=l2)
    work = len(X_tr) * epochs * net.n_params
    capped = False
    if work > MAX_WORK:
        epochs = max(1, int(MAX_WORK / (len(X_tr) * net.n_params)))
        capped = True
    opt = Optimizer(opt_kind, lr, [w.shape for w in net.W] + [b.shape for b in net.b])
    rng = np.random.default_rng(seed)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "grad_norm": []}
    diverged_at = None
    snapshots = []
    snap_every = max(1, epochs // 6)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X_tr_s))
        gn = 0.0
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            P, cache = net.forward(X_tr_s[bi], train=True)
            gW, gb = net.backward(cache, Y_tr[bi])
            gn = float(np.sqrt(sum(np.sum(g ** 2) for g in gW)))
            opt.step(net.W + net.b, gW + gb)
        P_tr = net.predict_proba(X_tr_s); P_va = net.predict_proba(X_va_s)
        tl, vl = net.loss(P_tr, Y_tr), net.loss(P_va, Y_va)
        finite = np.isfinite(tl) and np.isfinite(vl) and all(np.isfinite(w).all() for w in net.W)
        hist["epoch"].append(ep)
        hist["train_loss"].append(float(tl) if finite else None)
        hist["val_loss"].append(float(vl) if finite else None)
        hist["train_acc"].append(float((P_tr.argmax(1) == y_tr).mean()) if finite else None)
        hist["val_acc"].append(float((P_va.argmax(1) == y_va).mean()) if finite else None)
        hist["grad_norm"].append(gn if np.isfinite(gn) else None)
        if not finite or (tl > 50 and ep > 2):
            diverged_at = ep
            break
        if len(X_tr[0]) == 2 and (ep % snap_every == 0 or ep == 1):
            snapshots.append({"epoch": ep, "grid": _grid(net, scaler, X, res=30)})

    out = {"history": _clean_hist(hist), "classes": classes, "features": feats, "sizes": sizes, "n_params": net.n_params,
           "epochs_run": len(hist["epoch"]), "epochs_requested": int(config.get("epochs", 100)), "capped": capped,
           "diverged_at": diverged_at, "n_train": int(len(X_tr)), "n_val": int(len(X_va))}
    if diverged_at is None:
        fin = {k2: v[-1] for k2, v in hist.items() if k2 != "epoch"}
        out["final"] = {k2: (round(v, 4) if v is not None else None) for k2, v in fin.items()}
        best = int(np.nanargmin([v if v is not None else np.inf for v in hist["val_loss"]]))
        out["best_val_epoch"] = best + 1
        if X.shape[1] == 2:
            out["boundary"] = _grid(net, scaler, X, res=50)
            samp = np.random.default_rng(0).choice(len(X), min(400, len(X)), replace=False)
            out["points"] = {"x": X[samp, 0].round(4).tolist(), "y": X[samp, 1].round(4).tolist(), "label": y[samp].tolist()}
            out["snapshots"] = snapshots
    else:
        out["final"] = None
    out["weights"] = [np.round(W, 3).tolist() if W.size <= 2048 and np.isfinite(W).all() else None for W in net.W]
    out["weight_stats"] = [{"mean_abs": float(np.nanmean(np.abs(W))) if np.isfinite(W).all() else None, "shape": list(W.shape)} for W in net.W]
    out["diagnosis"] = diagnose(out, config)
    out["code"] = pytorch_code(dict(lr=lr, l2=l2, optimizer=opt_kind, dropout=dropout, epochs=epochs, batch_size=batch), sizes, act)
    out["train_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
    return out


def _clean_hist(h):
    return {k: [None if v is None else round(float(v), 5) for v in vals] for k, vals in h.items()}


def _grid(net, scaler, X, res=50):
    pad0 = (X[:, 0].max() - X[:, 0].min()) * .1
    pad1 = (X[:, 1].max() - X[:, 1].min()) * .1
    g0 = np.linspace(X[:, 0].min() - pad0, X[:, 0].max() + pad0, res)
    g1 = np.linspace(X[:, 1].min() - pad1, X[:, 1].max() + pad1, res)
    G = np.c_[np.tile(g0, res), np.repeat(g1, res)]  # x varies fastest → z[row=y][col=x]
    Gs = scaler.transform(G) if scaler is not None else G
    P = net.predict_proba(Gs)
    z = P[:, 1] if P.shape[1] == 2 else P.argmax(1) / max(P.shape[1] - 1, 1)
    z = np.nan_to_num(z, nan=0.5)
    return {"x_range": [float(g0[0]), float(g0[-1])], "y_range": [float(g1[0]), float(g1[-1])], "res": res,
            "z": np.round(z, 3).reshape(res, res).tolist()}


def diagnose(out, cfg) -> list[dict]:
    d = []
    h = out["history"]
    if out["diverged_at"]:
        d.append({"level": "danger", "code": "diverged", "text": f"Training DIVERGED at epoch {out['diverged_at']}: the loss became huge or NaN. The learning rate ({cfg.get('lr')}) is almost certainly too large for this optimizer — steps overshoot the valley and grow each time."})
        return d
    tl = [v for v in h["train_loss"] if v is not None]
    if len(tl) > 5:
        wiggle = np.mean(np.abs(np.diff(tl[-10:]))) / (np.mean(tl[-10:]) + 1e-9)
        drop = (tl[0] - tl[-1]) / (tl[0] + 1e-9)
        if wiggle > 0.15:
            d.append({"level": "warn", "code": "unstable", "text": "The loss oscillates strongly from epoch to epoch — a sign the learning rate is on the high side (or batches are tiny)."})
        if drop < 0.08:
            d.append({"level": "warn", "code": "slow", "text": f"Loss barely decreased ({drop:.0%} drop). The learning rate may be too small, the model too simple, or you need more epochs."})
    f = out["final"]
    if f and f["train_acc"] is not None and f["val_acc"] is not None:
        if f["train_acc"] - f["val_acc"] > 0.1:
            d.append({"level": "warn", "code": "overfit", "text": f"Overfitting: train accuracy {f['train_acc']:.2f} vs validation {f['val_acc']:.2f}. Try dropout, L2, fewer neurons, or early stopping (best validation loss was at epoch {out['best_val_epoch']})."})
        if f["train_acc"] < 0.75 and f["train_acc"] - f["val_acc"] < 0.05:
            d.append({"level": "info", "code": "underfit", "text": "Both accuracies are low: underfitting. Add capacity (more neurons/layers), use a non-linear activation, or train longer."})
        vl = [v for v in h["val_loss"] if v is not None]
        if len(vl) > 10 and out["best_val_epoch"] < 0.6 * len(vl) and vl[-1] > min(vl) * 1.15:
            d.append({"level": "info", "code": "early_stop", "text": f"Validation loss bottomed out at epoch {out['best_val_epoch']} and rose afterwards — early stopping would help."})
    if cfg.get("activation") == "linear" and len(cfg.get("hidden", [])) > 0:
        d.append({"level": "info", "code": "linear", "text": "All hidden activations are linear, so the whole network is equivalent to one linear model — it can only draw straight boundaries."})
    if out["capped"]:
        d.append({"level": "info", "code": "capped", "text": f"Epochs were capped at {out['epochs_run']} to keep the lab responsive."})
    if not d:
        d.append({"level": "ok", "code": "healthy", "text": "Training looks healthy: loss decreased smoothly and train/validation scores are close."})
    return d


def pytorch_code(cfg, sizes, act):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(f"    nn.Linear({sizes[i]}, {sizes[i + 1]}),")
        if i < len(sizes) - 2:
            layers.append(f"    nn.{ {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'linear': 'Identity'}[act] }(),")
            if cfg.get("dropout", 0) > 0:
                layers.append(f"    nn.Dropout(p={cfg.get('dropout')}),")
    opt = {"sgd": f"torch.optim.SGD(model.parameters(), lr={cfg.get('lr')}, weight_decay={cfg.get('l2', 0.0)})",
           "momentum": f"torch.optim.SGD(model.parameters(), lr={cfg.get('lr')}, momentum=0.9, weight_decay={cfg.get('l2', 0.0)})",
           "adam": f"torch.optim.Adam(model.parameters(), lr={cfg.get('lr')}, weight_decay={cfg.get('l2', 0.0)})"}[cfg.get("optimizer", "adam")]
    return "\n".join([
        "# Equivalent PyTorch code (the lab itself runs a transparent NumPy implementation)",
        "import torch, torch.nn as nn", "", "model = nn.Sequential(", *layers, ")",
        f"optimizer = {opt}", "loss_fn = nn.CrossEntropyLoss()", "",
        f"for epoch in range({cfg.get('epochs', 100)}):",
        "    model.train()",
        f"    for xb, yb in DataLoader(train_ds, batch_size={cfg.get('batch_size', 32)}, shuffle=True):",
        "        optimizer.zero_grad()",
        "        loss = loss_fn(model(xb), yb)   # forward pass",
        "        loss.backward()                 # backpropagation",
        "        optimizer.step()                # gradient-descent update",
        "    model.eval()  # disables dropout for validation",
    ])


SOURCE_EXCERPT = None


def source() -> str:
    import inspect
    return inspect.getsource(MLP) + "\n\n" + inspect.getsource(Optimizer)
