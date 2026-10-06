"""End-to-end game logic through the HTTP API (FastAPI TestClient + temporary SQLite)."""
import pytest

from neural_forge import bosses, game, missions, predictions
from neural_forge.curriculum import CONCEPTS


def answer_key(cid, q):
    return game._item_from_key(cid, q["key"])["answer"]


def ask(c, pid, cid, purpose="practice"):
    r = c.get(f"/api/p/{pid}/lesson/{cid}/question", params={"purpose": purpose})
    assert r.status_code == 200, r.text
    return r.json()


def answer(c, pid, cid, q, correct=True, purpose="practice"):
    resp = answer_key(cid, q) if correct else (-1 if q["type"] == "mcq" else 1e9)
    r = c.post(f"/api/p/{pid}/lesson/{cid}/answer", json={"key": q["key"], "response": resp, "purpose": purpose})
    assert r.status_code == 200, r.text
    return r.json()


def test_meta_and_player(client):
    m = client.get("/api/meta").json()
    assert len(m["modes"]) == 5 and len(m["concepts"]) >= 100 and len(m["datasets"]) >= 10
    r = client.post("/api/players", json={"name": "Ada", "mode": 3}).json()
    assert r["player"]["mode"] == 3 and r["rank"]["title"] == "AI Beginner"
    assert r["recommendations"], "a new player is told where to start"


def test_prerequisite_gating(player):
    c, pid = player
    tree = c.get(f"/api/p/{pid}/tree").json()
    locked = [n for n in tree["nodes"] if n["status"] == "locked"]
    assert locked and any(n["status"] == "new" for n in tree["nodes"])
    r = c.get(f"/api/p/{pid}/lesson/{locked[0]['id']}")
    assert r.status_code == 403


def test_lesson_teaches_first_and_mastery_unlocks(player):
    c, pid = player
    lesson = c.get(f"/api/p/{pid}/lesson/what_is_ai").json()
    assert lesson["plan"]["cards"][0] == "teach"
    unlocked = []
    for i in range(8):
        res = answer(c, pid, "what_is_ai", ask(c, pid, "what_is_ai"))
        assert res["correct"]
        unlocked += [u["id"] if isinstance(u, dict) else u for u in res["unlocked"]]
    assert res["status"] == "mastered"
    dependants = [cid for cid, cc in CONCEPTS.items() if cc["prereqs"] == ["what_is_ai"]]
    assert set(dependants) <= set(unlocked)
    ov = c.get(f"/api/p/{pid}").json()
    assert ov["player"]["xp"] > 0
    assert c.get(f"/api/p/{pid}/lesson/what_is_ai").json()["plan"]["kind"] == "quick_check"


def test_struggle_triggers_remediation(player):
    c, pid = player
    a = answer(c, pid, "what_is_ai", ask(c, pid, "what_is_ai"), correct=False)
    b = answer(c, pid, "what_is_ai", ask(c, pid, "what_is_ai"), correct=False)
    assert b["struggling"] and b["adaptation"]
    assert a["hint"], "Beginner mode shows a hint automatically after a miss"
    plan = c.get(f"/api/p/{pid}/lesson/what_is_ai").json()["plan"]
    assert plan["kind"] == "remediation" and plan["cards"][0] == "mini_lesson"
    q = ask(c, pid, "what_is_ai")
    assert q["difficulty"] == 1
    review = c.get(f"/api/p/{pid}/review").json()
    assert review == [] or all(r["id"] for r in review)


def test_research_mode_has_no_hints(client):
    pid = client.post("/api/players", json={"name": "R", "mode": 5}).json()["player"]["id"]
    q = client.get(f"/api/p/{pid}/lesson/what_is_ai/question", params={"purpose": "guided"}).json()
    assert not q["hint_available"] and "hint" not in q


@pytest.mark.parametrize("exp_id", list(predictions.EXPERIMENTS)[:4])
def test_predictions_are_computed(player, exp_id):
    c, pid = player
    r = c.post(f"/api/p/{pid}/predict/{exp_id}", json={"choice": 0}).json()
    assert r["was_right"] == (r["correct_option"] == 0)
    assert r["explanation"] and r["data"]


def test_ml_run_history_and_compare(player):
    c, pid = player
    base = dict(dataset="student_success", target="passed", features=["study_hours", "attendance", "previous_grade"], seed=1)
    a = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(base, model="logistic_regression"), "name": "lr"}).json()
    b = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(base, model="random_forest")}).json()
    assert a["ok"] and b["ok"]
    runs = c.get(f"/api/p/{pid}/runs").json()
    assert {r["id"] for r in runs} >= {a["run_id"], b["run_id"]}
    for r in runs:  # reproducibility record
        assert {"dataset", "features", "target", "model", "params", "preprocessing", "seed"} <= set(r["config"])
        assert r["ts"]
    assert c.post(f"/api/p/{pid}/runs/{a['run_id']}", json={"notes": "baseline"}).json()["ok"]
    cmp = c.post(f"/api/p/{pid}/compare", json={"run_ids": [a["run_id"], b["run_id"]]}).json()
    assert len(cmp["rows"]) == 2 and cmp["insights"]
    bad = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(base, features=[], model="logistic_regression")}).json()
    assert bad["ok"] is False and bad["lesson"]


def test_full_mission(player):
    c, pid = player
    m = missions.BY_ID["m_student"]
    mid = m["id"]
    assert c.post(f"/api/p/{pid}/missions/{mid}", json={"action": "data_answer", "payload": {"answer": 4}}).json()["correct"]
    c.post(f"/api/p/{pid}/missions/{mid}", json={"action": "hypothesis", "payload": {"text": "study hours matter", "bucket": 2}})
    cfg = dict(dataset=m["dataset"], target="passed", **m["recommended"])
    run = c.post(f"/api/p/{pid}/run/ml", json={"config": cfg}).json()
    sub = c.post(f"/api/p/{pid}/missions/{mid}", json={"action": "submit_run", "payload": {"run_id": run["run_id"]}}).json()
    assert sub["check"]["passed"]
    done = c.post(f"/api/p/{pid}/missions/{mid}", json={"action": "reflect", "payload": {"answer": 1}}).json()
    assert done["completed"] and done["xp"] >= m["xp"]


def test_banned_feature_fails_mission(player):
    c, pid = player
    run = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(dataset="student_success", target="passed", features=["student_id", "study_hours", "attendance", "previous_grade"], model="logistic_regression")}).json()
    sub = c.post(f"/api/p/{pid}/missions/m_student", json={"action": "submit_run", "payload": {"run_id": run["run_id"]}}).json()
    assert not sub["check"]["passed"]


def test_overfitter_boss_full_fight(player):
    c, pid = player
    b = c.get(f"/api/p/{pid}/bosses/overfitter").json()
    assert b["state"]["hp"] == 100 and "answer" not in b["phases"][0]["q"], "answers are never sent to the client"
    wrong = c.post(f"/api/p/{pid}/bosses/overfitter", json={"answer": 0}).json()
    assert not wrong["evaluation"]["passed"] and wrong["state"]["mistakes"] == 1
    assert c.post(f"/api/p/{pid}/bosses/overfitter", json={"answer": 1}).json()["evaluation"]["passed"]
    # the naive fix (deep tree on everything) must FAIL
    feats_all = [f for f in __import__("neural_forge.datasets", fromlist=["load"]).load("overfit_lab").columns if f != "label"]
    naive = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(dataset="overfit_lab", target="label", features=feats_all, model="decision_tree")}).json()
    assert not c.post(f"/api/p/{pid}/bosses/overfitter", json={"run_id": naive["run_id"]}).json()["evaluation"]["passed"]
    good = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(dataset="overfit_lab", target="label", features=["signal_a", "signal_b", "noise_00"], model="logistic_regression")}).json()
    assert c.post(f"/api/p/{pid}/bosses/overfitter", json={"run_id": good["run_id"]}).json()["evaluation"]["passed"]
    fin = c.post(f"/api/p/{pid}/bosses/overfitter", json={"answer": 3}).json()
    assert fin["defeated"] and fin["xp"] == 400 + 100 - 20 * 2  # two mistakes: wrong diagnosis + naive run


def test_leak_boss_select_accepts_labels_or_indices(player):
    c, pid = player
    opts = bosses.BY_ID["leak"]["phases"][0]["options"]
    idx = [opts.index(x) for x in bosses.LEAKY]
    assert c.post(f"/api/p/{pid}/bosses/leak", json={"selected": idx}).json()["evaluation"]["passed"]
    c.post(f"/api/p/{pid}/bosses/leak", json={"restart": True})
    assert c.post(f"/api/p/{pid}/bosses/leak", json={"selected": sorted(bosses.LEAKY)}).json()["evaluation"]["passed"]


def test_injection_boss_requires_code_level_defence(player):
    c, pid = player
    c.post(f"/api/p/{pid}/bosses/injection", json={"answer": bosses.BY_ID["injection"]["phases"][0]["q"]["answer"]})
    weak = c.post(f"/api/p/{pid}/run/agent", json={"config": {"defences": ["injection_classifier"]}}).json()
    assert not c.post(f"/api/p/{pid}/bosses/injection", json={"run_id": weak["run_id"]}).json()["evaluation"]["passed"]
    strong = c.post(f"/api/p/{pid}/run/agent", json={"config": {"defences": ["separate_channels", "random_delimiters", "confirm_sensitive"]}}).json()
    assert c.post(f"/api/p/{pid}/bosses/injection", json={"run_id": strong["run_id"]}).json()["evaluation"]["passed"]


def test_code_exercise_grading(player):
    c, pid = player
    bad = c.post(f"/api/p/{pid}/exercises/ex_variables", json={"code": "hours = 7\nrate = 2\ntotal = 0"}).json()
    assert not bad["passed"]
    good = c.post(f"/api/p/{pid}/exercises/ex_variables", json={"code": "hours = 7\nrate = 2.5\ntotal = hours * rate"}).json()
    assert good["passed"] and good["xp"] > 0
    assert c.get(f"/api/p/{pid}/exercises/ex_variables").json()["solution"], "solution revealed after solving"


def test_research_challenge_scoring(player):
    c, pid = player
    run = c.post(f"/api/p/{pid}/run/ml", json={"config": dict(dataset="house_prices", target="price", features=["size_sqm", "bedrooms", "bathrooms", "age_years", "distance_km", "has_garage", "neighborhood"], model="linear_regression")}).json()
    sc = c.post(f"/api/p/{pid}/challenges/rc_house_budget", json={"run_id": run["run_id"]}).json()
    assert sc["valid"] and 0 < sc["score"] <= 100 and sc["improved"]
    wrong = c.post(f"/api/p/{pid}/challenges/rc_fraud_cost", json={"run_id": run["run_id"]}).json()
    assert wrong["valid"] is False


@pytest.mark.parametrize("name", ["linreg", "split", "cv", "overfit", "threshold", "kmeans", "conv", "tokenizer", "attention", "lm", "chunks", "boundary",
                                  "distribution", "histogram", "onehot", "scaling", "imbalance", "missing", "peek"])
def test_widgets(client, name):
    r = client.get(f"/api/widget/{name}")
    assert r.status_code == 200, r.text


def test_unknown_things_404(player):
    c, pid = player
    assert c.get("/api/p/9999").status_code == 404
    assert c.get(f"/api/p/{pid}/bosses/nope").status_code == 404
    assert c.get(f"/api/p/{pid}/missions/nope").status_code == 404
