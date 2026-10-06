"""Metrics must be REAL: recompute them independently and check the pipeline's honesty guards."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from neural_forge import datalab, datasets, ml


def run(**kw):
    ds = kw.setdefault("dataset", "student_success")
    info = datasets.REGISTRY[ds]
    kw.setdefault("target", info["target"])
    kw.setdefault("features", [c for c in datasets.load(ds).columns if c != info["target"]])
    return ml.run_experiment(ml.Experiment.from_dict(kw))


def test_all_datasets_load_and_profile():
    for d in datasets.catalog():
        df = datasets.load(d["id"])
        assert len(df) >= 100, d["id"]
        p = datalab.profile(df, d["target"])
        assert p["rows"] == len(df)


def test_metrics_are_consistent_with_confusion_matrix():
    r = run(model="logistic_regression", features=["study_hours", "attendance", "previous_grade", "sleep_hours"], preprocessing={"scaling": "standard"})
    cm = np.array(r["confusion_matrix"]["matrix"])
    assert cm.sum() == r["n_test"]
    acc = np.trace(cm) / cm.sum()
    assert r["metrics"]["test"]["accuracy"] == pytest.approx(acc, abs=1e-3)
    tn, fp, fn, tp = cm.ravel()
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert r["metrics"]["test"]["f1"] == pytest.approx(f1, abs=1e-3)


def test_metrics_match_an_independent_sklearn_run():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = datasets.load("moons")
    X, y = df[["x1", "x2"]], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict(Xte)
    r = run(dataset="moons", model="logistic_regression", features=["x1", "x2"], preprocessing={"scaling": "standard"})
    assert r["metrics"]["test"]["accuracy"] == pytest.approx(accuracy_score(yte, ref), abs=1e-3)
    assert r["metrics"]["test"]["f1"] == pytest.approx(f1_score(yte, ref), abs=1e-3)


def test_reproducible_with_same_seed():
    a = run(model="random_forest", seed=7)
    b = run(model="random_forest", seed=7)
    assert a["metrics"] == b["metrics"]


def test_unlimited_tree_overfits_overfit_lab():
    r = run(dataset="overfit_lab", model="decision_tree")
    assert r["metrics"]["train"]["accuracy"] == 1.0
    assert r["metrics"]["test"]["accuracy"] < 0.8
    assert any(d["code"] == "overfit" for d in r["diagnosis"])


def test_leakage_is_too_good_to_be_true_and_removing_it_is_honest():
    from neural_forge.bosses import LEAKY, LOAN_ALL
    leaky = run(dataset="loan_leak", model="random_forest", features=LOAN_ALL)
    clean = run(dataset="loan_leak", model="logistic_regression", features=[f for f in LOAN_ALL if f not in LEAKY])
    assert leaky["metrics"]["test"]["accuracy"] > 0.97
    assert 0.65 < clean["metrics"]["test"]["accuracy"] < 0.9


def test_majority_baseline_on_imbalanced_fraud():
    r = run(dataset="fraud", model="dummy")
    assert r["metrics"]["test"]["accuracy"] > 0.97 and r["metrics"]["test"]["recall"] == 0.0


def test_regression_and_clustering():
    r = run(dataset="house_prices", model="linear_regression")
    assert r["task"] == "regression" and r["metrics"]["test"]["r2"] > 0.8
    assert r["metrics"]["test"]["rmse"] < r["baseline"]["rmse"]
    c = run(dataset="customer_segments", target=None, model="kmeans", params={"n_clusters": 5}, preprocessing={"scaling": "standard"})
    assert c["task"] == "clustering" and -1 <= c["metrics"]["train"]["silhouette"] <= 1
    inertias = [e["inertia"] for e in c["elbow"]]
    assert all(a >= b - 1e-6 for a, b in zip(inertias, inertias[1:])), "inertia must decrease with k"


def test_pipeline_errors_teach():
    with pytest.raises(ml.PipelineError) as e:
        run(model="logistic_regression", features=[])
    assert e.value.to_dict()["lesson"]
    with pytest.raises(ml.PipelineError):
        run(dataset="house_prices", model="logistic_regression")  # continuous target for a classifier
    with pytest.raises(ml.PipelineError):
        run(dataset="data_chaos", model="logistic_regression")  # '34 yrs' text in numeric column without fix_types


def test_generated_code_mentions_the_model():
    r = run(model="random_forest", params={"n_estimators": 50})
    assert "RandomForestClassifier" in r["code"] and "n_estimators=50" in r["code"]


def test_text_classification():
    r = run(dataset="spam", model="naive_bayes", features=["text"])
    assert r["metrics"]["test"]["f1"] > 0.8
