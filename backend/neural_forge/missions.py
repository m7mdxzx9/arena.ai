"""Dataset missions and open-ended research challenges.

Mission loop: briefing → inspect data (Data Lab) → data question (computed from the real dataset) →
hypothesis (which feature matters most?) → prediction (expected score bucket) → experiment(s) in the Workbench
→ objectives checked server-side against the stored run → reflection → XP + mastery evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import datasets
from .curriculum.schema import Q

OPS = {">=": lambda a, b: a is not None and a >= b, "<=": lambda a, b: a is not None and a <= b}


def M(id, title, area, dataset, task, concepts, requires, briefing, objectives, *, xp=200, banned=(), banned_reason="",
      data_question="majority", metric_buckets=None, reflection=None, recommended=None, mentor_tip="", extra=None):
    return dict(id=id, title=title, area=area, dataset=dataset, task=task, concepts=concepts, requires=requires, briefing=briefing,
                objectives=objectives, xp=xp, banned=list(banned), banned_reason=banned_reason, data_question=data_question,
                metric_buckets=metric_buckets, reflection=reflection, recommended=recommended or {}, mentor_tip=mentor_tip, extra=extra or {})


def O(metric, op, value, label=None, split="test"):
    return dict(metric=metric, op=op, value=value, split=split, label=label or f"Test {metric.upper()} {op} {value}")


MISSIONS = [
    M("m_student", "Early Warning System", "ml_workshop", "student_success", "classification",
      ["classification", "features", "logistic_regression", "train_test_split"], ["classification"],
      "The Academy wants to offer tutoring BEFORE students fail. Build a classifier that flags students at risk. Careful: a column in this dataset carries no real information.",
      [O("f1", ">=", 0.72)], banned=["student_id"], banned_reason="student_id is an arbitrary identifier — a model can memorise it but it can't generalise.",
      data_question="missing", metric_buckets=["< 0.50", "0.50 – 0.70", "0.70 – 0.85", "> 0.85"],
      reflection=Q(2, "Your model flags at-risk students. Which error is worse for this use case?",
                   ["Flagging a student who would have passed (false positive: some extra tutoring)", "Missing a student who will fail (false negative: no help offered)"], 1,
                   "Missing a struggling student means no support; an extra tutoring offer is cheap. That argues for prioritising recall."),
      recommended=dict(model="logistic_regression", features=["study_hours", "attendance", "previous_grade", "sleep_hours"], preprocessing=dict(scaling="standard")),
      mentor_tip="Start with a simple, interpretable model. You can always add complexity later."),
    M("m_house", "Fair Price Estimator", "ml_workshop", "house_prices", "regression",
      ["regression", "linear_regression", "mae", "rmse", "r2"], ["regression"],
      "The housing cooperative needs price estimates members can trust. Predict sale prices with R² of at least 0.90 on unseen houses.",
      [O("r2", ">=", 0.90, "Test R² ≥ 0.90"), O("mae", "<=", 20000, "Test MAE ≤ €20,000")], data_question="target_median",
      metric_buckets=["R² < 0.5", "0.5 – 0.8", "0.8 – 0.93", "> 0.93"],
      reflection=Q(2, "Your MAE is about €16k and RMSE about €20k. What does RMSE > MAE tell you?",
                   ["All errors are exactly equal", "Some predictions miss by much more than average", "The model is overfitting", "R² is negative"], 1,
                   "RMSE squares errors before averaging, so a few big misses pull it above MAE."),
      recommended=dict(model="linear_regression", features=["size_sqm", "bedrooms", "bathrooms", "age_years", "distance_km", "has_garage", "neighborhood"]),
      mentor_tip="Neighborhood is categorical — one-hot encoding lets a linear model give each area its own premium."),
    M("m_spam", "Phishing Shield", "data_district", "spam", "text_classification",
      ["features", "precision", "tokenization"], ["classification", "features"],
      "Students are drowning in phishing. Build a filter — but deleting real messages is unacceptable, so precision matters most.",
      [O("precision", ">=", 0.90, "Test precision ≥ 0.90"), O("f1", ">=", 0.85, "Test F1 ≥ 0.85")], data_question="majority",
      metric_buckets=["F1 < 0.6", "0.6 – 0.8", "0.8 – 0.92", "> 0.92"],
      reflection=Q(2, "5% of labels in this dataset are deliberately wrong. What does that imply?",
                   ["A perfect model would get 100% test accuracy", "Even a perfect model can't exceed roughly 95% accuracy against these labels", "Labels don't matter", "Use a bigger model"], 1,
                   "Label noise caps measurable performance — chasing 100% would mean learning the mistakes."),
      recommended=dict(model="naive_bayes", features=["text"]), mentor_tip="Text needs to become numbers: TF-IDF counts words, weighting rare informative ones higher."),
    M("m_churn", "Keep the Subscribers", "evaluation_chamber", "customer_churn", "classification",
      ["recall", "class_imbalance", "categorical", "encoding"], ["recall"],
      "The campus ISP can call customers who are about to leave. Missing a churner costs a subscriber; an unnecessary call costs little. Catch at least 65% of churners while keeping F1 ≥ 0.60.",
      [O("recall", ">=", 0.65, "Test recall ≥ 0.65"), O("f1", ">=", 0.60, "Test F1 ≥ 0.60")], data_question="strongest_corr",
      metric_buckets=["recall < 0.4", "0.4 – 0.6", "0.6 – 0.8", "> 0.8"],
      reflection=Q(2, "Which lever raised recall without retraining a different algorithm?",
                   ["Lowering the decision threshold or using class_weight='balanced'", "Adding the customer's name", "Using fewer test rows", "Training for more epochs"], 0,
                   "Both shift the trade-off toward predicting the positive class more often."),
      recommended=dict(model="logistic_regression", features=["tenure_months", "contract", "monthly_charges", "support_calls", "internet", "paperless_billing"],
                       preprocessing=dict(scaling="standard"), params=dict(class_weight="balanced")),
      mentor_tip="Try class_weight='balanced' or lower the threshold slider."),
    M("m_digits", "First Light", "vision_lab", "digits", "classification",
      ["classification", "knn", "cnn_intro"], ["knn"],
      "The Vision Lab's first camera sees only 8×8 pixels. Teach a model to recognise digits 0–9 with ≥ 95% accuracy.",
      [O("accuracy", ">=", 0.95, "Test accuracy ≥ 0.95")], data_question="n_classes",
      metric_buckets=["< 0.6", "0.6 – 0.85", "0.85 – 0.95", "> 0.95"],
      reflection=Q(2, "Each pixel is a feature here. Why would a CNN usually beat k-NN on larger images?",
                   ["CNN filters detect local patterns (edges, strokes) wherever they appear", "CNNs don't need data", "k-NN can't do multiclass", "CNNs ignore pixels"], 0,
                   "Weight sharing + local receptive fields capture spatial structure that raw-pixel distance ignores."),
      recommended=dict(model="knn", features=[f"px_{i}" for i in range(64)]), mentor_tip="Similar digits have similar pixel patterns — nearest neighbours is a natural first try."),
    M("m_animals", "Zoo Records", "ml_workshop", "animals", "classification",
      ["decision_trees", "classification"], ["decision_trees"],
      "The campus zoo lost its labels. Classify animals into five classes with a model the zookeepers can read (macro F1 ≥ 0.88).",
      [O("f1", ">=", 0.88, "Test macro-F1 ≥ 0.88")], data_question="n_classes",
      metric_buckets=["< 0.6", "0.6 – 0.8", "0.8 – 0.92", "> 0.92"],
      reflection=Q(2, "Macro-F1 averages per-class F1 equally. Why use it here?",
                   ["Rare classes (like reptiles) count as much as common ones", "It is always higher", "It ignores mistakes", "It's required for trees"], 0,
                   "Plain accuracy would be dominated by the most common class (mammals)."),
      recommended=dict(model="decision_tree", features=["weight_kg", "height_cm", "legs", "has_fur", "lays_eggs", "can_fly", "aquatic"], params=dict(max_depth=5)),
      mentor_tip="A shallow tree is a readable flowchart. Check the importances afterwards."),
    M("m_medical", "Clinic Triage (synthetic)", "evaluation_chamber", "medical", "classification",
      ["recall", "precision", "f1"], ["recall", "precision"],
      "A fictional clinic wants a screening aid. All data is synthetic — this is about metric trade-offs, not medical advice. Catch ≥ 70% of cases with precision ≥ 0.45.",
      [O("recall", ">=", 0.70, "Test recall ≥ 0.70"), O("precision", ">=", 0.45, "Test precision ≥ 0.45")], data_question="majority",
      metric_buckets=["recall < 0.4", "0.4 – 0.6", "0.6 – 0.8", "> 0.8"],
      reflection=Q(3, "A screening tool should be used how?",
                   ["To replace clinicians", "To prioritise cases for human review, with monitoring for errors", "Without any evaluation", "Only on training data"], 1,
                   "High-stakes AI supports human decisions; it needs ongoing evaluation and accountability."),
      recommended=dict(model="logistic_regression", features=["age", "bmi", "blood_pressure", "glucose", "cholesterol", "smoker", "activity_level"],
                       preprocessing=dict(scaling="standard"), params=dict(class_weight="balanced")),
      mentor_tip="Balanced class weights trade some precision for recall."),
    M("m_segments", "Know Your Customers", "data_district", "customer_segments", "clustering",
      ["clustering", "scaling", "unsupervised"], ["clustering"],
      "The campus store wants customer groups for tailored offers — but nobody knows the groups. Find well-separated clusters (silhouette ≥ 0.45) using 3–6 clusters.",
      [O("silhouette", ">=", 0.45, "Silhouette ≥ 0.45", split="train"), dict(metric="n_clusters", op="between", value=[3, 6], label="3 ≤ k ≤ 6", split="config")],
      data_question="scale_gap", metric_buckets=["< 0.2", "0.2 – 0.4", "0.4 – 0.55", "> 0.55"],
      reflection=Q(2, "You found 5 clusters. What should you do before the store uses them?",
                   ["Nothing — the algorithm is always right", "Inspect cluster profiles and check they make business sense", "Delete small clusters", "Re-run until silhouette = 1"], 1,
                   "Clusters are hypotheses. Domain experts must interpret them."),
      recommended=dict(model="kmeans", features=["annual_income_k", "age", "spending_score", "visits_per_month"], preprocessing=dict(scaling="standard"), params=dict(n_clusters=5)),
      mentor_tip="Look at the elbow chart, and scale the features — income would dominate otherwise."),
    M("m_sentiment", "Review Radar", "language_center", "sentiment", "text_classification",
      ["tokenization", "embeddings", "classification"], ["tokenization"],
      "Which products disappoint customers? Classify review sentiment with F1 ≥ 0.70 — and discover what bag-of-words can't understand.",
      [O("f1", ">=", 0.70, "Test F1 ≥ 0.70")], data_question="majority",
      metric_buckets=["< 0.55", "0.55 – 0.7", "0.7 – 0.85", "> 0.85"],
      reflection=Q(3, "15% of reviews start with 'not', which flips their meaning. Why does a bag-of-words model struggle with them?",
                   ["It ignores word order and context, so 'not great' looks positive", "It can't count words", "It needs more epochs", "Reviews are too short"], 0,
                   "Order-aware models (sequence models, transformers) use context. Bigrams help a little: 'not great' becomes one feature."),
      recommended=dict(model="logistic_regression", features=["review"], params=dict(C=10)), mentor_tip="TF-IDF with bigrams can capture short phrases. Try weaker regularisation (larger C) and compare TF-IDF with plain word counts."),
    M("m_fraud", "Reliable Fraud Report", "evaluation_chamber", "fraud", "classification",
      ["cross_validation", "class_imbalance", "validation"], ["cross_validation"],
      "The card office wants a fraud model AND an honest estimate of how stable its score is. Use ≥ 5-fold cross-validation and reach test F1 ≥ 0.45.",
      [O("f1", ">=", 0.45, "Test F1 ≥ 0.45"), dict(metric="cv_folds", op=">=", value=5, label="Cross-validation with ≥ 5 folds", split="config")],
      data_question="majority", metric_buckets=["F1 < 0.2", "0.2 – 0.4", "0.4 – 0.6", "> 0.6"],
      reflection=Q(3, "Your CV F1 is 0.52 ± 0.09. How should you report it?",
                   ["'F1 = 0.52' — the std is noise", "'F1 ≈ 0.52 ± 0.09 across 5 folds' — fold-to-fold variation is large with few fraud cases", "'F1 = 0.61', the best fold", "Don't report it"], 1,
                   "With only ~100 fraud cases, fold scores vary; reporting uncertainty is responsible evaluation."),
      recommended=dict(model="random_forest", features=["amount", "hour", "distance_from_home_km", "foreign", "card_present", "tx_last_hour"],
                       threshold=0.3, cv_folds=5),
      mentor_tip="Set CV folds to 5 in the Workbench. Look at the spread, not just the mean."),
]
BY_ID = {m["id"]: m for m in MISSIONS}


def data_question(mission: dict) -> dict:
    """Build a question whose answer is computed from the actual dataset."""
    df = datasets.load(mission["dataset"])
    target = datasets.REGISTRY[mission["dataset"]]["target"]
    kind = mission["data_question"]
    if kind == "missing":
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        opts = [c for c in df.columns if c != target][:6]
        ans = miss.index[0]
        if ans not in opts:
            opts[-1] = ans
        return dict(prompt="Inspect the Data Lab. Which column has the MOST missing values?", options=opts, answer=opts.index(ans),
                    explanation=f"'{ans}' has {int(miss.iloc[0])} missing values ({miss.iloc[0] / len(df):.1%}). You'll need an imputation strategy for it.")
    if kind == "majority":
        share = df[target].value_counts(normalize=True).iloc[0]
        cands = sorted({round(share * 100), 50, 75, 90, 98} - {round(share * 100)})
        opts = [f"{round(share * 100)}%"] + [f"{c}%" for c in cands[:3]]
        opts.sort(key=lambda s: int(s[:-1]))
        return dict(prompt="What share of rows belong to the most common class of the target?", options=opts, answer=opts.index(f"{round(share * 100)}%"),
                    explanation=f"The majority class is {share:.1%} of the data. A model that always predicts it gets {share:.1%} accuracy — your baseline to beat.")
    if kind == "target_median":
        med = float(df[target].median())
        opts_v = [med, med * 0.6, med * 1.5, med * 2.2]
        opts = [f"€{round(v / 1000) * 1000:,.0f}" for v in opts_v]
        correct = opts[0]
        opts.sort()
        return dict(prompt="Roughly what is the MEDIAN sale price?", options=opts, answer=opts.index(correct),
                    explanation=f"Median price ≈ €{med:,.0f}. Knowing the target's scale helps interpret MAE: an error of €20k is ~{20000 / med:.0%} of a typical house.")
    if kind == "strongest_corr":
        num = df.select_dtypes("number").drop(columns=[target])
        corr = num.corrwith(df[target]).abs().sort_values(ascending=False)
        opts = corr.index[:4].tolist()
        ans = opts[0]
        import random
        random.Random(1).shuffle(opts)
        return dict(prompt="Which numeric feature has the strongest (absolute) correlation with the target?", options=opts, answer=opts.index(ans),
                    explanation=f"'{ans}' (|r| = {corr.iloc[0]:.2f}). Correlation only captures linear relationships between numeric columns — categorical columns like contract type may matter too.")
    if kind == "n_classes":
        n = df[target].nunique()
        opts = sorted({n, 2, n + 3, max(3, n - 2)})
        return dict(prompt="How many distinct classes does the target have?", options=[str(o) for o in opts], answer=opts.index(n),
                    explanation=f"{n} classes → multiclass classification. Metrics like F1 are averaged across classes (macro).")
    if kind == "scale_gap":
        stds = df.std().sort_values(ascending=False)
        opts = stds.index.tolist()
        return dict(prompt="Which feature has the LARGEST spread (standard deviation)? It will dominate distances unless you scale.", options=opts, answer=opts.index(stds.index[0]),
                    explanation=f"'{stds.index[0]}' (std {stds.iloc[0]:.1f}) vs '{stds.index[-1]}' (std {stds.iloc[-1]:.1f}). k-means uses distances, so scale first.")
    raise ValueError(kind)


def metric_value(run: dict, obj: dict):
    res, cfg = run["result"], run["config"]
    if obj["split"] == "config":
        if obj["metric"] == "n_clusters":
            return (cfg.get("params") or {}).get("n_clusters", 3)
        if obj["metric"] == "cv_folds":
            return cfg.get("cv_folds", 0)
    return res["metrics"].get(obj["split"], {}).get(obj["metric"])


def check_objectives(mission: dict, run: dict, bonus: float = 0.0) -> dict:
    cfg = run["config"]
    checks = []
    if cfg.get("dataset") != mission["dataset"]:
        return dict(passed=False, checks=[dict(label="Use the mission dataset", passed=False, value=cfg.get("dataset"))])
    used_banned = [f for f in cfg.get("features", []) if f in mission["banned"]]
    if mission["banned"]:
        checks.append(dict(label=f"Don't use {', '.join(mission['banned'])}", passed=not used_banned, value=used_banned or "ok", why=mission["banned_reason"]))
    for o in mission["objectives"]:
        v = metric_value(run, o)
        if o["op"] == "between":
            ok = v is not None and o["value"][0] <= v <= o["value"][1]
        else:
            target = o["value"] + (bonus if o["op"] == ">=" and isinstance(o["value"], float) and o["value"] <= 1 else 0)
            ok = OPS[o["op"]](v, target)
        checks.append(dict(label=o["label"], passed=bool(ok), value=v))
    return dict(passed=all(c["passed"] for c in checks), checks=checks)


def bucket_index(mission: dict, run: dict) -> int | None:
    o = mission["objectives"][0]
    v = metric_value(run, o)
    if v is None or not mission.get("metric_buckets"):
        return None
    import re
    edges = []
    for b in mission["metric_buckets"][1:]:
        nums = re.findall(r"\d+\.\d+|\d+", b)
        edges.append(float(nums[0]))
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


# ------------------------------------------------------------------ Research challenges (open-ended, scored)
def _cost(run, fn_cost=50, fp_cost=2):
    cm = run["result"]["confusion_matrix"]["matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return fn * fn_cost + fp * fp_cost, dict(tn=tn, fp=fp, fn=fn, tp=tp)


def score_challenge(ch_id: str, run: dict) -> dict:
    res, cfg = run["result"], run["config"]
    if ch_id == "rc_fraud_cost":
        if cfg["dataset"] != "fraud":
            return dict(valid=False, reason="Use the fraud dataset.")
        cost, cells = _cost(run)
        n_pos = cells["fn"] + cells["tp"]
        baseline = n_pos * 50  # never flag anything
        score = round(100 * max(0.0, 1 - cost / baseline), 1)
        return dict(valid=True, score=score, details=dict(cost=cost, baseline_cost=baseline, **cells),
                    explanation=f"Cost = 50 × missed frauds + 2 × false alarms = {cost} (doing nothing costs {baseline}). Score = % of cost saved.")
    if ch_id == "rc_house_budget":
        if cfg["dataset"] != "house_prices":
            return dict(valid=False, reason="Use the house prices dataset.")
        rmse, base = res["metrics"]["test"]["rmse"], res["baseline"]["rmse"]
        lat = res.get("predict_ms_per_1k") or 0
        interp = res.get("interpretability", 0)
        acc_score = max(0, 1 - rmse / base) * 70
        speed_score = 15 * min(1.0, 3 / max(lat, 1e-6))
        interp_score = interp * 3
        score = round(acc_score + speed_score + interp_score, 1)
        return dict(valid=True, score=score, details=dict(rmse=rmse, baseline_rmse=base, predict_ms_per_1k=lat, interpretability=interp),
                    explanation="Score = 70 × (1 − RMSE/baseline RMSE) + up to 15 for latency (full marks ≤ 3 ms per 1,000 predictions) + 3 × interpretability (1–5). Accuracy isn't everything.")
    if ch_id == "rc_stable_student":
        if cfg["dataset"] != "student_success":
            return dict(valid=False, reason="Use the student success dataset.")
        cv = res.get("cv")
        if not cv or cv["folds"] < 5:
            return dict(valid=False, reason="Run with ≥ 5 cross-validation folds.")
        if "student_id" in cfg.get("features", []):
            return dict(valid=False, reason="student_id is not allowed.")
        score = round(100 * (cv["mean"] - 2 * cv["std"]), 1)
        return dict(valid=True, score=score, details=dict(cv_mean=cv["mean"], cv_std=cv["std"]),
                    explanation="Score = 100 × (CV mean F1 − 2 × CV std). Rewards models that are good AND stable across folds.")
    raise KeyError(ch_id)


CHALLENGES = [
    dict(id="rc_fraud_cost", title="The Cost of Fraud", dataset="fraud", requires=["class_imbalance"],
         brief="Each missed fraud costs €50; each false alarm costs €2 (a customer call). Minimise total cost on the test set. Thresholds, class weights, models — your choice. There is no single right answer."),
    dict(id="rc_house_budget", title="Price Engine on a Budget", dataset="house_prices", requires=["rmse"],
         brief="Deploy a price model on a tiny server for a cooperative that must explain prices to members. You're scored on accuracy, prediction latency AND interpretability."),
    dict(id="rc_stable_student", title="Reproducible Early Warning", dataset="student_success", requires=["cross_validation"],
         brief="A model that scores 0.80 on one split and 0.60 on another can't be trusted. Maximise CV mean F1 − 2 × std (≥ 5 folds)."),
]
