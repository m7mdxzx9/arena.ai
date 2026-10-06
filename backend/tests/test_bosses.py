"""Every boss is beatable with a principled fix, and the naive 'fixes' do not work."""
import pytest

from neural_forge import agent, bosses, ml, nn, rag
from neural_forge.curriculum import CONCEPTS

B = bosses.BY_ID


def runml(cfg):
    return ml.run_experiment(ml.Experiment.from_dict(cfg))


def phase(bid, idx, result=None, cfg=None, payload=None):
    return bosses.evaluate_phase(B[bid], idx, payload or {}, result, cfg, 0.0)["passed"]


def test_bosses_are_well_formed():
    assert len(bosses.BOSSES) == 8
    for b in bosses.BOSSES:
        assert all(r in CONCEPTS for r in b["requires"])
        pub = bosses.public(b)
        for ph in pub["phases"]:
            assert "answer" not in ph and "answer" not in ph.get("q", {})


def test_mcq_phases_have_valid_answers():
    for b in bosses.BOSSES:
        for i, ph in enumerate(b["phases"]):
            if ph["kind"] == "mcq":
                assert phase(b["id"], i, payload={"answer": ph["q"]["answer"]})


def test_overfitter():
    feats = bosses._overfit_feats()
    naive = dict(dataset="overfit_lab", target="label", features=feats, model="decision_tree", params=dict(max_depth=3))
    assert not phase("overfitter", 1, runml(naive), naive)
    good = dict(dataset="overfit_lab", target="label", features=["signal_a", "signal_b", "noise_00"], model="logistic_regression")
    assert phase("overfitter", 1, runml(good), good)


def test_leak():
    leaky = dict(dataset="loan_leak", target="defaulted", features=bosses.LOAN_ALL, model="logistic_regression")
    assert not phase("leak", 1, runml(leaky), leaky)
    honest = dict(leaky, features=[f for f in bosses.LOAN_ALL if f not in bosses.LEAKY])
    assert phase("leak", 1, runml(honest), honest)


def test_imbalance():
    naive = dict(dataset="fraud", target="is_fraud", features=bosses.FRAUD, model="random_forest")
    assert not phase("imbalance", 1, runml(naive), naive)
    good = dict(naive, threshold=0.2, preprocessing=dict(scaling="standard"))
    assert phase("imbalance", 1, runml(good), good)


def test_data_chaos():
    full = dict(dedupe=True, fix_types=True, normalize_categories=True, invalid_to_missing=True, clip_outliers=True, scaling="standard")
    good = dict(dataset="data_chaos", target="cancelled", features=bosses.CHAOS, model="logistic_regression", preprocessing=full)
    assert phase("chaos", 1, runml(good), good)
    partial = dict(good, preprocessing=dict(full, dedupe=False))
    assert not phase("chaos", 1, runml(partial), partial)
    assert phase("chaos", 0, payload={"selected": B["chaos"]["phases"][0]["answer"]})
    assert not phase("chaos", 0, payload={"selected": ["Target leakage"]})


def test_learning_rate_beast():
    slow = dict(dataset="spiral", hidden=[32, 32], lr=0.0005, optimizer="sgd", epochs=100)
    assert phase("lr_beast", 1, nn.train(slow), slow)
    tame = dict(dataset="spiral", hidden=[32, 32], lr=0.01, optimizer="adam", epochs=200)
    assert phase("lr_beast", 2, nn.train(tame), tame)
    wild = dict(tame, optimizer="sgd", lr=8.0, epochs=60)
    assert not phase("lr_beast", 2, nn.train(wild), wild)


def test_hallucination():
    closed = dict(mode="closed_book")
    assert not phase("hallucination", 1, rag.evaluate_answers(closed), closed)
    good = dict(mode="grounded", embedding="lsa", chunk_size=60, method="hybrid", top_k=3, abstain_threshold=0.5)
    assert phase("hallucination", 1, rag.evaluate_answers(good), good)


def test_retrieval_failure():
    broken = dict(chunk_size=300, overlap=0, embedding="hash32", method="dense", top_k=1, context_budget=80)
    assert not phase("retrieval", 1, rag.evaluate_retrieval(broken), broken)
    good = dict(chunk_size=40, embedding="lsa", method="hybrid", top_k=2, rerank=True, context_budget=80)
    assert phase("retrieval", 1, rag.evaluate_retrieval(good), good)


@pytest.mark.parametrize("defences,ok", [([], False), (["injection_classifier"], False), (["separate_channels"], False),
                                         (["separate_channels", "random_delimiters", "confirm_sensitive"], True)])
def test_prompt_injection(defences, ok):
    assert phase("injection", 1, agent.evaluate(defences), {"defences": defences}) is ok
