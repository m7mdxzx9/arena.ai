"""Parametric question generators.

Each generator returns a freshly-built question whose correct answer is *computed*
(never hand-typed), so practice is unlimited and difficulty can scale with mastery.
Signature: gen(rng: random.Random, difficulty: int) -> question dict.
"""
from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np


def _num(d, prompt, answer, tol, why, hint=""):
    return {"type": "numeric", "difficulty": d, "prompt": prompt, "answer": float(answer), "tolerance": tol,
            "explanation": why, "hint": hint, "generated": True}


def _mcq_from_value(d, prompt, correct, distractors, why, rng, hint=""):
    opts = [str(correct)] + [str(x) for x in distractors if str(x) != str(correct)]
    opts = list(dict.fromkeys(opts))[:4]
    rng.shuffle(opts)
    return {"type": "mcq", "difficulty": d, "prompt": prompt, "options": opts, "answer": opts.index(str(correct)),
            "explanation": why, "hint": hint, "generated": True}


def _run(code: str) -> dict:
    """Execute *our own generated* snippet to compute the true answer."""
    ns: dict = {}
    exec(code, {"__builtins__": {"range": range, "len": len, "sum": sum, "print": lambda *a: None}}, ns)
    return ns


def py_trace_vars(rng, d):
    a, b = rng.randint(2, 9), rng.randint(2, 6)
    if d == 1:
        code = f"x = {a}\nx = x + {b}"
    elif d == 2:
        code = f"x = {a}\ny = x * {b}\nx = y - x"
    else:
        code = f"x = {a}\ny = {b}\nx, y = y, x + y\nx = x * y"
    ans = _run(code)["x"]
    return _num(d, f"What is the value of x after running:\n```python\n{code}\n```", ans, 0.001,
                f"Trace line by line. Final x = {ans}.", "Evaluate the right-hand side using current values first.")


def py_trace_if(rng, d):
    s = rng.randint(30, 100)
    t1, t2 = (90, 70) if d < 3 else (rng.randint(75, 95), rng.randint(50, 74))
    code = f"score = {s}\nif score >= {t1}:\n    r = 3\nelif score >= {t2}:\n    r = 2\nelse:\n    r = 1"
    if d >= 2:
        code += f"\nif score % 2 == 0:\n    r = r * 10"
    ans = _run(code)["r"]
    return _num(d, f"What is r?\n```python\n{code}\n```", ans, 0.001, f"score={s}: r = {ans}.",
                "Only the first true branch of an if/elif chain runs.")


def py_trace_loop(rng, d):
    n = rng.randint(3, 6) if d == 1 else rng.randint(4, 9)
    if d == 1:
        code = f"total = 0\nfor i in range({n}):\n    total = total + i"
    elif d == 2:
        k = rng.randint(2, 3)
        code = f"total = 0\nfor i in range({n}):\n    if i % {k} == 0:\n        total = total + i"
    else:
        code = f"total = 0\nfor i in range(1, {n}):\n    for j in range(i):\n        total = total + 1"
    ans = _run(code)["total"]
    return _num(d, f"What is total?\n```python\n{code}\n```", ans, 0.001, f"Tracing every iteration gives {ans}.",
                "range(n) goes from 0 to n-1.")


def py_trace_list(rng, d):
    xs = [rng.randint(1, 9) for _ in range(5)]
    if d == 1:
        i = rng.randint(0, 4)
        code, expr = f"a = {xs}", f"a[{i}]"
    elif d == 2:
        code, expr = f"a = {xs}", "a[-2]"
    else:
        i, j = sorted(rng.sample(range(6), 2))
        code, expr = f"a = {xs}\nb = a[{i}:{j}]", "sum(b)"
    ns = _run(code + f"\nresult = {expr}")
    return _num(d, f"```python\n{code}\n```\nWhat is `{expr}`?", ns["result"], 0.001,
                f"Answer: {ns['result']}. Remember indices start at 0, negatives count from the end, slices exclude the end.")


def np_shape(rng, d):
    r, c = rng.randint(2, 50), rng.randint(2, 10)
    if d == 1:
        return _mcq_from_value(d, f"A dataset of {r} samples and {c} features. NumPy shape?", (r, c), [(c, r), (r * c,), (r,)],
                               "Rows (samples) first, then columns (features).", rng)
    if d == 2:
        return _mcq_from_value(d, f"X.shape == ({r}, {c}). What is X.mean(axis=0).shape?", (c,), [(r,), (r, c), ()],
                               "axis=0 averages over rows, leaving one value per column.", rng)
    k = rng.randint(2, 8)
    return _mcq_from_value(d, f"X.shape == ({r}, {c}), W.shape == ({c}, {k}). Shape of X @ W?", (r, k), [(c, k), (r, c), (k, r)],
                           "Inner dims cancel: (r×c)·(c×k) = r×k. This is a dense layer!", rng)


def alg_linear(rng, d):
    a, b, x = rng.randint(2, 9), rng.randint(-10, 10), rng.randint(-5, 10)
    if d == 1:
        return _num(d, f"y = {a}x + ({b}). If x = {x}, y = ?", a * x + b, 0.001, f"{a}·{x} + ({b}) = {a * x + b}.")
    y = a * x + b
    return _num(d, f"Solve for x: {a}x + ({b}) = {y}", x, 0.001, f"x = ({y} − ({b})) / {a} = {x}.")


def vec_dot(rng, d):
    n = 2 if d == 1 else (3 if d == 2 else 4)
    a = [rng.randint(-4, 5) for _ in range(n)]
    b = [rng.randint(-4, 5) for _ in range(n)]
    if d == 3:
        if not any(a) or not any(b):
            a[0] = b[0] = 1
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return _num(d, f"Cosine similarity of a={a} and b={b}? (2 dp)", round(cos, 2), 0.011,
                    f"a·b / (|a||b|) = {np.dot(a, b)} / ({np.linalg.norm(a):.2f}·{np.linalg.norm(b):.2f}) ≈ {cos:.2f}.")
    return _num(d, f"Dot product of a={a} and b={b}?", int(np.dot(a, b)), 0.001, f"Σ aᵢbᵢ = {int(np.dot(a, b))}.")


def mat_shape(rng, d):
    r, k, c = rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6)
    if d < 3:
        return _mcq_from_value(d, f"Shape of ({r}×{k}) · ({k}×{c})?", f"{r}×{c}", [f"{k}×{k}", f"{c}×{r}", f"{r}×{k}"],
                               "Outer dimensions survive.", rng)
    M = np.array([[rng.randint(-3, 4) for _ in range(2)] for _ in range(2)])
    v = np.array([rng.randint(-3, 4) for _ in range(2)])
    out = M @ v
    return _num(d, f"M = {M.tolist()}, v = {v.tolist()}. First element of M·v?", int(out[0]), 0.001, f"Row 1 · v = {int(out[0])}.")


def prob_basic(rng, d):
    if d == 1:
        p = round(rng.uniform(0.05, 0.95), 2)
        return _num(d, f"P(event) = {p}. P(not event)?", round(1 - p, 2), 0.001, "Complement rule: 1 − p.")
    if d == 2:
        p, q = round(rng.uniform(0.1, 0.9), 1), round(rng.uniform(0.1, 0.9), 1)
        return _num(d, f"Independent events A (p={p}) and B (p={q}). P(A and B)?", round(p * q, 3), 0.001, "Multiply for independent events.")
    prev = rng.choice([0.01, 0.02, 0.05]); sens = rng.choice([0.9, 0.95, 0.99]); spec = rng.choice([0.9, 0.95, 0.99])
    post = prev * sens / (prev * sens + (1 - prev) * (1 - spec))
    return _num(d, f"Prevalence {prev}, test sensitivity {sens}, specificity {spec}. P(disease | positive)? (2 dp)",
                round(post, 2), 0.011, f"Bayes: {prev}·{sens} / ({prev}·{sens} + {1 - prev:.2f}·{1 - spec:.2f}) ≈ {post:.2f}.")


def stats_mean_median(rng, d):
    xs = sorted(rng.randint(1, 20) for _ in range(5 if d < 3 else 6))
    if d >= 2:
        xs[-1] = rng.randint(60, 200)
    if d == 1:
        return _num(d, f"Mean of {xs}?", round(float(np.mean(xs)), 2), 0.011, f"Sum {sum(xs)} / {len(xs)}.")
    return _num(d, f"Median of {xs}?", float(np.median(xs)), 0.001,
                f"Middle value(s) of the sorted list = {np.median(xs)}. (Mean = {np.mean(xs):.1f}, pulled up by the outlier.)")


def deriv_slope(rng, d):
    a, x = rng.randint(1, 5), rng.randint(-4, 5)
    if d == 1:
        return _num(d, f"f(x) = {a}x². Derivative is {2 * a}x. Slope at x = {x}?", 2 * a * x, 0.001, f"{2 * a}·{x}.")
    b = rng.randint(-5, 5)
    return _num(d, f"f(x) = {a}x² + ({b})x. Slope at x = {x}? (derivative: {2 * a}x + ({b}))", 2 * a * x + b, 0.001,
                f"{2 * a}·{x} + ({b}) = {2 * a * x + b}.")


def gd_step(rng, d):
    w = round(rng.uniform(-3, 3), 1); g = round(rng.uniform(-4, 4), 1); lr = rng.choice([0.01, 0.1, 0.5])
    if d < 3:
        return _num(d, f"w = {w}, gradient = {g}, learning rate = {lr}. New w after one gradient-descent step?",
                    round(w - lr * g, 4), 0.001, f"w − lr·g = {w} − {lr}·({g}) = {w - lr * g:.4f}.")
    # two steps on L(w) = (w - t)^2
    t = rng.randint(-3, 3)
    w1 = w - lr * 2 * (w - t)
    w2 = w1 - lr * 2 * (w1 - t)
    return _num(d, f"Loss L(w) = (w − {t})², gradient 2(w − {t}). Start w = {w}, lr = {lr}. w after TWO steps? (3 dp)",
                round(w2, 3), 0.002, f"Step 1: {w1:.3f}; Step 2: {w2:.3f}.")


def acc_from_cm(rng, d):
    tp, fp, fn, tn = (rng.randint(5, 60) for _ in range(4))
    if d == 3:
        tn = rng.randint(400, 900)
        tp, fp, fn = rng.randint(1, 10), rng.randint(1, 10), rng.randint(10, 40)
    acc = (tp + tn) / (tp + fp + fn + tn)
    why = f"(TP+TN)/total = ({tp}+{tn})/{tp + fp + fn + tn} = {acc:.3f}."
    if d == 3:
        why += f" Note recall is only {tp / (tp + fn):.2f} — high accuracy can hide many misses."
    return _num(d, f"TP={tp}, FP={fp}, FN={fn}, TN={tn}. Accuracy? (3 dp)", round(acc, 3), 0.002, why)


def prec_rec(rng, d):
    tp, fp, fn = rng.randint(5, 80), rng.randint(1, 50), rng.randint(1, 50)
    which = rng.choice(["precision", "recall"])
    val = tp / (tp + fp) if which == "precision" else tp / (tp + fn)
    formula = "TP/(TP+FP)" if which == "precision" else "TP/(TP+FN)"
    return _num(d, f"TP={tp}, FP={fp}, FN={fn}. Compute {which}. (3 dp)", round(val, 3), 0.002, f"{which} = {formula} = {val:.3f}.",
                "Precision: of predicted positives. Recall: of actual positives.")


def f1_calc(rng, d):
    p, r = round(rng.uniform(0.1, 1.0), 2), round(rng.uniform(0.1, 1.0), 2)
    f1 = 2 * p * r / (p + r)
    return _num(d, f"Precision = {p}, recall = {r}. F1? (3 dp)", round(f1, 3), 0.002, f"2PR/(P+R) = {f1:.3f}.")


def reg_metrics(rng, d):
    y = [rng.randint(10, 50) for _ in range(4)]
    p = [v + rng.randint(-6, 6) for v in y]
    err = np.array(y) - np.array(p)
    which = ["mae", "mse", "rmse"][min(d, 3) - 1]
    val = {"mae": np.abs(err).mean(), "mse": (err ** 2).mean(), "rmse": math.sqrt((err ** 2).mean())}[which]
    return _num(d, f"Actual = {y}, predicted = {p}. Compute {which.upper()}. (2 dp)", round(float(val), 2), 0.011,
                f"Errors = {err.tolist()} → {which.upper()} = {val:.2f}.")


def neuron_forward(rng, d):
    x = [rng.randint(-2, 3) for _ in range(2 if d == 1 else 3)]
    w = [round(rng.uniform(-1, 1), 1) + 0.0 for _ in x]
    b = round(rng.uniform(-1, 1), 1) + 0.0
    z = float(np.dot(x, w) + b)
    if d < 3:
        return _num(d, f"Inputs {x}, weights {w}, bias {b}. ReLU output? (2 dp)", round(max(0, z), 2), 0.011,
                    f"z = {z:.2f}; ReLU(z) = {max(0, z):.2f}.")
    s = 1 / (1 + math.exp(-z))
    return _num(d, f"Inputs {x}, weights {w}, bias {b}. Sigmoid output? (3 dp)", round(s, 3), 0.002, f"z = {z:.2f}; σ(z) = {s:.3f}.")


def mrr_calc(rng, d):
    ranks = [rng.randint(1, 5) for _ in range(3 if d < 3 else 5)]
    mrr = float(np.mean([1 / r for r in ranks]))
    return _num(d, f"Rank of first relevant chunk per query: {ranks}. MRR? (3 dp)", round(mrr, 3), 0.002, f"Mean of 1/rank = {mrr:.3f}.")


GENERATORS: dict[str, Callable] = {f.__name__: f for f in [
    py_trace_vars, py_trace_if, py_trace_loop, py_trace_list, np_shape, alg_linear, vec_dot, mat_shape, prob_basic,
    stats_mean_median, deriv_slope, gd_step, acc_from_cm, prec_rec, f1_calc, reg_metrics, neuron_forward, mrr_calc]}


def generate(name: str, difficulty: int, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    return GENERATORS[name](rng, difficulty)
