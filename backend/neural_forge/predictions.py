"""Predict-before-running experiments.

The player commits to a prediction, then two REAL experiments run. The correct option is decided by
the actual outcome (`judge`) — not hard-coded — so if a run surprises us, the game says so honestly.
"""
from __future__ import annotations

from . import ml, nn, rag

SS_FEATS = ["study_hours", "attendance", "previous_grade", "sleep_hours"]
FRAUD_FEATS = ["amount", "hour", "distance_from_home_km", "foreign", "card_present", "tx_last_hour"]
LOAN_OK = ["income_k", "debt_ratio", "credit_score", "term_months", "late_payments_before_loan"]


def _ml(**kw):
    return ml.run_experiment(ml.Experiment.from_dict(kw))


def _lr_explosion():
    a = nn.train(dict(dataset="moons", hidden=[16, 16], lr=0.05, optimizer="sgd", epochs=80, seed=1))
    b = nn.train(dict(dataset="moons", hidden=[16, 16], lr=8.0, optimizer="sgd", epochs=80, seed=1))
    unstable = bool(b["diverged_at"]) or any(d["code"] in ("unstable", "diverged") for d in b["diagnosis"]) or \
        (b["final"] and a["final"] and b["final"]["val_acc"] < a["final"]["val_acc"] - 0.05)
    correct = 1 if unstable else (0 if b["final"]["val_acc"] >= a["final"]["val_acc"] else 2)
    return correct, dict(kind="curves", runs=[dict(label="lr = 0.05", history=a["history"], final=a["final"], diverged_at=a["diverged_at"]),
                                                dict(label="lr = 8.0", history=b["history"], final=b["final"], diverged_at=b["diverged_at"])])


def _tree_depth():
    a = _ml(dataset="student_success", target="passed", features=SS_FEATS, model="decision_tree", params=dict(max_depth=3))
    b = _ml(dataset="student_success", target="passed", features=SS_FEATS, model="decision_tree", params=dict(max_depth=None))
    ga = a["metrics"]["train"]["accuracy"] - a["metrics"]["test"]["accuracy"]
    gb = b["metrics"]["train"]["accuracy"] - b["metrics"]["test"]["accuracy"]
    correct = 1 if gb > ga + 0.03 and b["metrics"]["train"]["accuracy"] >= a["metrics"]["train"]["accuracy"] else (0 if b["metrics"]["test"]["accuracy"] > a["metrics"]["test"]["accuracy"] else 2)
    return correct, dict(kind="bars", metric="accuracy", runs=[
        dict(label="max_depth = 3", train=a["metrics"]["train"]["accuracy"], test=a["metrics"]["test"]["accuracy"]),
        dict(label="max_depth = None", train=b["metrics"]["train"]["accuracy"], test=b["metrics"]["test"]["accuracy"])])


def _imbalance():
    a = _ml(dataset="fraud", target="is_fraud", features=FRAUD_FEATS, model="dummy")
    acc = a["metrics"]["test"]["accuracy"]
    correct = min(range(4), key=lambda i: abs([0.5, 0.75, 0.98, 0.10][i] - acc))
    return correct, dict(kind="table", rows=[dict(metric="accuracy", value=acc), dict(metric="recall (fraud caught)", value=a["metrics"]["test"]["recall"]),
                                             dict(metric="precision", value=a["metrics"]["test"]["precision"]), dict(metric="F1", value=a["metrics"]["test"]["f1"])],
                         confusion=a["confusion_matrix"])


def _scaling():
    feats = ["tenure_months", "monthly_charges", "support_calls", "paperless_billing"]
    a = _ml(dataset="customer_churn", target="churned", features=feats, model="knn", params=dict(n_neighbors=15))
    b = _ml(dataset="customer_churn", target="churned", features=feats, model="knn", params=dict(n_neighbors=15), preprocessing=dict(scaling="standard"))
    fa, fb = a["metrics"]["test"]["f1"], b["metrics"]["test"]["f1"]
    correct = 0 if fb > fa + 0.02 else (1 if fb < fa - 0.02 else 2)
    return correct, dict(kind="bars", metric="F1", runs=[dict(label="k-NN, raw features", test=fa, train=a["metrics"]["train"]["f1"]),
                                                         dict(label="k-NN, standardised", test=fb, train=b["metrics"]["train"]["f1"])])


def _knn_k():
    a = _ml(dataset="moons", target="label", features=["x1", "x2"], model="knn", params=dict(n_neighbors=1))
    b = _ml(dataset="moons", target="label", features=["x1", "x2"], model="knn", params=dict(n_neighbors=40))
    correct = 0 if a["metrics"]["train"]["accuracy"] > b["metrics"]["train"]["accuracy"] else 1
    return correct, dict(kind="boundaries", runs=[dict(label="k = 1", boundary=a["decision_boundary"], train=a["metrics"]["train"]["accuracy"], test=a["metrics"]["test"]["accuracy"]),
                                                  dict(label="k = 40", boundary=b["decision_boundary"], train=b["metrics"]["train"]["accuracy"], test=b["metrics"]["test"]["accuracy"])])


def _missing():
    try:
        _ml(dataset="student_success", target="passed", features=SS_FEATS, model="logistic_regression", preprocessing=dict(impute_numeric="none"))
        outcome, correct = "Training succeeded.", 0
    except ml.PipelineError as e:
        outcome, correct = e.message, 1
    b = _ml(dataset="student_success", target="passed", features=SS_FEATS, model="logistic_regression", preprocessing=dict(impute_numeric="median"))
    return correct, dict(kind="table", rows=[dict(metric="No imputation", value=outcome),
                                             dict(metric="Median imputation → test accuracy", value=b["metrics"]["test"]["accuracy"])])


def _leak():
    allf = LOAN_OK + ["collection_calls_after_due", "days_since_last_payment_at_audit"]
    a = _ml(dataset="loan_leak", target="defaulted", features=allf, model="random_forest")
    b = _ml(dataset="loan_leak", target="defaulted", features=LOAN_OK, model="random_forest")
    da = a["metrics"]["test"]["accuracy"] - b["metrics"]["test"]["accuracy"]
    correct = 1 if da > 0.05 else (0 if da < -0.02 else 2)
    return correct, dict(kind="bars", metric="accuracy", runs=[dict(label="With leaky columns", train=a["metrics"]["train"]["accuracy"], test=a["metrics"]["test"]["accuracy"]),
                                                               dict(label="Without leaky columns", train=b["metrics"]["train"]["accuracy"], test=b["metrics"]["test"]["accuracy"])])


def _kmeans():
    feats = ["annual_income_k", "age", "spending_score", "visits_per_month"]
    a = _ml(dataset="customer_segments", target=None, features=feats, model="kmeans", params=dict(n_clusters=2), preprocessing=dict(scaling="standard"))
    b = _ml(dataset="customer_segments", target=None, features=feats, model="kmeans", params=dict(n_clusters=8), preprocessing=dict(scaling="standard"))
    ia, ib = a["metrics"]["train"]["inertia"], b["metrics"]["train"]["inertia"]
    correct = 0 if ib < ia else 1
    return correct, dict(kind="elbow", elbow=a["elbow"], runs=[dict(label="k = 2", inertia=ia, silhouette=a["metrics"]["train"]["silhouette"]),
                                                              dict(label="k = 8", inertia=ib, silhouette=b["metrics"]["train"]["silhouette"])])


def _dropout():
    base = dict(dataset="moons", hidden=[64, 64], lr=0.01, epochs=300, train_limit=30, batch_size=8, seed=42)
    a = nn.train({**base, "dropout": 0.0})
    b = nn.train({**base, "dropout": 0.5})
    ga = a["final"]["train_acc"] - a["final"]["val_acc"]; gb = b["final"]["train_acc"] - b["final"]["val_acc"]
    correct = 0 if (gb < ga - 0.01 or b["final"]["val_loss"] < a["final"]["val_loss"] * 0.85) else (1 if gb > ga + 0.01 else 2)
    return correct, dict(kind="curves", runs=[dict(label="dropout 0.0", history=a["history"], final=a["final"]),
                                              dict(label="dropout 0.5", history=b["history"], final=b["final"])])


def _capacity():
    a = nn.train(dict(dataset="spiral", hidden=[2], lr=0.01, epochs=200, seed=3))
    b = nn.train(dict(dataset="spiral", hidden=[32, 32], lr=0.01, epochs=200, seed=3))
    correct = 1 if b["final"]["val_acc"] > a["final"]["val_acc"] + 0.1 else 2
    return correct, dict(kind="nn_boundaries", runs=[dict(label="1 hidden layer × 2 neurons", boundary=a.get("boundary"), points=a.get("points"), final=a["final"]),
                                                     dict(label="2 hidden layers × 32 neurons", boundary=b.get("boundary"), points=b.get("points"), final=b["final"])])


def _activation():
    a = nn.train(dict(dataset="xor", hidden=[16, 16], activation="linear", lr=0.01, epochs=150, seed=0))
    b = nn.train(dict(dataset="xor", hidden=[16, 16], activation="relu", lr=0.01, epochs=150, seed=0))
    correct = 0 if b["final"]["val_acc"] > a["final"]["val_acc"] + 0.1 else 2
    return correct, dict(kind="nn_boundaries", runs=[dict(label="linear activations", boundary=a.get("boundary"), points=a.get("points"), final=a["final"]),
                                                     dict(label="ReLU activations", boundary=b.get("boundary"), points=b.get("points"), final=b["final"])])


def _chunk():
    a = rag.evaluate_retrieval(dict(chunk_size=300, overlap=0, embedding="tfidf", method="dense", top_k=2, context_budget=80))
    b = rag.evaluate_retrieval(dict(chunk_size=40, overlap=10, embedding="tfidf", method="dense", top_k=2, context_budget=80))
    ca, cb = a["metrics"]["context_hit_rate"], b["metrics"]["context_hit_rate"]
    correct = 1 if cb > ca + 0.02 else (0 if ca > cb + 0.02 else 2)
    return correct, dict(kind="table", rows=[dict(metric="300-word chunks: answer reached the context window", value=ca),
                                             dict(metric="40-word chunks: answer reached the context window", value=cb),
                                             dict(metric="300-word chunks: recall@2 (anywhere in retrieved chunk)", value=a["metrics"]["recall@2"]),
                                             dict(metric="40-word chunks: recall@2", value=b["metrics"]["recall@2"])])


EXPERIMENTS = {
    "lr_explosion": dict(concept="learning_rate", title="The Learning-Rate Dial",
                         setup="A small neural network (2 hidden layers × 16 neurons, plain SGD) is learning the 'two moons' dataset with learning rate 0.05. You turn the dial to 8.0.",
                         question="What do you think will happen?",
                         options=["Faster, stable convergence and better accuracy", "Unstable training: the loss jumps around or explodes", "No noticeable difference", "Guaranteed higher accuracy, just noisier"],
                         explain="Gradient descent moves each weight by lr × gradient. When steps are much larger than the width of the valley in the loss landscape, each update overshoots the minimum, so the loss bounces or grows. Bigger is not faster beyond a point.",
                         run=_lr_explosion),
    "tree_depth": dict(concept="overfitting", title="Grow the Tree",
                       setup="A decision tree predicts exam success. Currently max_depth = 3. You remove the limit (max_depth = None) so it can grow until every leaf is pure.",
                       question="What happens to training vs test accuracy?",
                       options=["Both rise together", "Training accuracy rises to ~100% while test accuracy stalls or drops (bigger gap)", "Both drop", "Nothing changes"],
                       explain="An unlimited tree can carve out a leaf for every training example — memorising noise. Training accuracy approaches 100%, but those tiny leaves don't generalise, so the train–test gap widens.",
                       run=_tree_depth),
    "imbalance_baseline": dict(concept="class_imbalance", title="The Lazy Fraud Detector",
                               setup="In the card-fraud dataset about 2% of transactions are fraud. A 'model' always predicts 'not fraud'.",
                               question="Roughly what test accuracy will this lazy model get?",
                               options=["About 50%", "About 75%", "About 98%", "About 10%"],
                               explain="Accuracy counts correct predictions. If 98% of cases are legitimate, saying 'legit' every time is right 98% of the time — while catching zero fraud. That's why recall, precision and F1 matter on imbalanced data.",
                               run=_imbalance),
    "scaling_knn": dict(concept="scaling", title="Units Matter",
                        setup="k-NN (k=15) predicts customer churn from tenure (1–72 months), monthly charges (€18–120), support calls (0–8) and paperless billing (0/1) — very different ranges. You add standard scaling.",
                        question="What happens to the test F1 score?",
                        options=["It improves", "It gets worse", "No meaningful change"],
                        explain="k-NN measures Euclidean distance. Unscaled, a €20 difference in monthly charges outweighs 3 extra support calls — even though support calls strongly predict churn. Standardising puts features on comparable scales so the distance reflects all of them.",
                        run=_scaling),
    "knn_k": dict(concept="knn", title="How Many Neighbours?",
                  setup="Two k-NN classifiers learn the two-moons data: k = 1 and k = 40.",
                  question="Which one has the HIGHER accuracy on its own training data?",
                  options=["k = 1", "k = 40"],
                  explain="With k = 1, each training point's nearest neighbour is itself, so training accuracy is 100% by construction — a perfect memoriser with a jagged boundary. k = 40 averages many neighbours: smoother boundary, lower train accuracy, often similar or better test accuracy.",
                  run=_knn_k),
    "missing_strategy": dict(concept="missing_values", title="Holes in the Data",
                             setup="The student dataset has missing sleep_hours and attendance values. You train logistic regression with NO imputation.",
                             question="What happens?",
                             options=["It trains normally and ignores the missing cells", "scikit-learn refuses to train and raises an error", "Missing values are silently treated as 0"],
                             explain="Most scikit-learn estimators raise 'Input X contains NaN'. You must decide explicitly: impute (mean/median/most frequent), drop, or use a model that supports missing values natively.",
                             run=_missing),
    "leak_removal": dict(concept="data_leakage", title="Plugging the Leak",
                         setup="A random forest predicts loan default and scores almost perfectly. Two columns — collection calls after due date and days since last payment at audit — are only known AFTER a default happens. You remove them.",
                         question="What happens to test accuracy?",
                         options=["It rises", "It drops noticeably — the 'perfect' score was fake", "It stays the same"],
                         explain="The leaky columns encode the answer. Removing them gives an honest score that reflects what the model can actually know at application time. A lower honest score beats a fake perfect one.",
                         run=_leak),
    "kmeans_k": dict(concept="clustering", title="More Clusters?",
                     setup="k-means groups customers. You compare k = 2 with k = 8.",
                     question="What happens to inertia (total distance of points to their centroid)?",
                     options=["It decreases with k = 8", "It increases with k = 8", "It stays the same"],
                     explain="More centroids means every point can be closer to one. Inertia always (weakly) falls as k grows — so minimum inertia can't pick k. Look for the 'elbow' or use silhouette.",
                     run=_kmeans),
    "dropout_gap": dict(concept="dropout", title="Random Silence",
                        setup="A big network (2 × 64 neurons) trains on only 30 two-moons points for 300 epochs. You add dropout 0.5.",
                        question="What happens to overfitting (train–validation gap / validation loss)?",
                        options=["Overfitting decreases", "Overfitting increases", "No change"],
                        explain="Dropout randomly silences neurons during training, so the network can't rely on fragile co-adapted neurons that memorise individual points. It acts as a regulariser — usually reducing the gap at some cost in training fit.",
                        run=_dropout),
    "capacity_spiral": dict(concept="neural_networks", title="Enough Neurons?",
                            setup="Two networks learn interleaved spirals: one with a single hidden layer of 2 neurons, one with two hidden layers of 32 neurons.",
                            question="Which statement will be true on the validation set?",
                            options=["The tiny network wins: simpler is always better", "The larger network is far more accurate", "Both reach about the same accuracy"],
                            explain="Two ReLU neurons can only combine two straight cuts — far too few to wrap around a spiral (underfitting). More neurons/layers give the capacity to bend the decision boundary.",
                            run=_capacity),
    "activation_linear": dict(concept="activation", title="Remove the Bends",
                              setup="Two identical networks (2 × 16 neurons) learn XOR. One uses linear activations, the other ReLU.",
                              question="What happens?",
                              options=["ReLU solves XOR; the linear network fails", "The linear network wins", "Both solve it equally"],
                              explain="A stack of linear layers is itself linear, so it can only draw one straight line — XOR needs two. Non-linear activations make depth meaningful.",
                              run=_activation),
    "chunk_size": dict(concept="chunking", title="Slice Size",
                       setup="A RAG system has an 80-word context budget and retrieves the top-2 chunks. You switch from 300-word chunks (whole documents) to 40-word chunks.",
                       question="How often will the answer actually reach the model's context window?",
                       options=["Less often with small chunks", "More often with small chunks", "Same"],
                       explain="Large chunks may be retrieved correctly, but the context budget truncates them — answers deep inside a long chunk get cut off. Smaller, focused chunks fit the budget and carry the relevant sentence.",
                       run=_chunk),
}


def catalog():
    return [dict(id=k, concept=v["concept"], title=v["title"], setup=v["setup"], question=v["question"], options=v["options"]) for k, v in EXPERIMENTS.items()]


def run(exp_id: str, choice: int) -> dict:
    e = EXPERIMENTS[exp_id]
    correct, data = e["run"]()
    return dict(id=exp_id, concept=e["concept"], choice=choice, correct_option=correct, was_right=(choice == correct),
                options=e["options"], explanation=e["explain"], data=data)
