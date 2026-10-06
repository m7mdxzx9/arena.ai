"""The adaptive-learning model is a documented Bayesian Knowledge Tracing variant — test its math and policies."""
import pytest

from neural_forge import mastery as m

MCQ = dict(type="mcq", options=["a", "b", "c", "d"], difficulty=1)


def test_bkt_correct_increases_and_wrong_decreases():
    assert m.bkt_update(0.3, True, guess=0.35, slip=0.1) > 0.3
    # a wrong answer lowers the posterior before learning-transit is applied
    no_transit = m.bkt_update(0.5, False, guess=0.35, slip=0.1, transit=0.0)
    assert no_transit < 0.5


def test_bkt_matches_closed_form():
    p, g, s, t = 0.4, 0.25, 0.1, 0.15
    post = p * (1 - s) / (p * (1 - s) + (1 - p) * g)
    assert m.bkt_update(p, True, guess=g, slip=s, transit=t) == pytest.approx(post + (1 - post) * t)


def test_guess_probabilities():
    assert m.guess_prob(MCQ) == pytest.approx(0.35)
    assert m.guess_prob(dict(type="mcq", options=["y", "n"])) == pytest.approx(0.5)  # capped
    assert m.guess_prob(dict(type="numeric")) == 0.05


def test_hint_weakens_evidence():
    s0 = m.update_state({}, MCQ, True, hint_used=False)
    s1 = m.update_state({}, MCQ, True, hint_used=True)
    assert s1["p"] < s0["p"]


def test_hard_correct_answer_is_stronger_evidence():
    easy = m.update_state({"p": 0.3}, dict(MCQ, difficulty=1), True)
    hard = m.update_state({"p": 0.3}, dict(MCQ, difficulty=3), True)
    assert hard["p"] > easy["p"]


def test_status_thresholds_and_min_correct():
    assert m.status(None, False) == "locked"
    assert m.status(None, True) == "new"
    assert m.status({"p": 0.5, "attempts": 2}, True) == "learning"
    assert m.status({"p": 0.7, "attempts": 2}, True) == "proficient"
    assert m.status({"p": 0.9, "attempts": 2, "correct": 2}, True) == "proficient"  # needs ≥3 correct
    assert m.status({"p": 0.9, "attempts": 3, "correct": 3}, True) == "mastered"


def test_struggle_detection_and_remediation_policy():
    st = {}
    st = m.update_state(st, dict(MCQ, difficulty=2), False)
    assert not st.get("struggling")
    st = m.update_state(st, dict(MCQ, difficulty=2), False)
    assert st["struggling"], "two wrong in a row → struggling"
    assert m.status(st, True) == "struggling"
    assert m.target_difficulty(st, 3) == 1, "struggling → easiest questions"
    plan = m.lesson_plan(st, 3)
    assert plan["kind"] == "remediation" and plan["cards"][0] == "mini_lesson" and "practice_easy" in plan["cards"]
    # recovery: two easy correct → not struggling, retry the failed difficulty next
    st = m.update_state(st, MCQ, True)
    st = m.update_state(st, MCQ, True)
    assert not st["struggling"] and st["retry_pending"]
    assert m.target_difficulty(st, 3) == 2
    st = m.update_state(st, dict(MCQ, difficulty=2), True)
    assert not st["retry_pending"]


def test_mastery_makes_lessons_shorter_and_harder():
    full = m.lesson_plan(None, 2)
    assert full["cards"][0] == "teach"
    assert m.lesson_plan({"p": 0.7, "attempts": 3}, 2)["kind"] == "practice"
    quick = m.lesson_plan({"p": 0.95, "attempts": 5, "correct": 5}, 2)
    assert quick["kind"] == "quick_check" and set(quick["cards"]) == {"challenge"}
    assert "code" in m.lesson_plan(None, 4)["cards"], "Engineer mode adds code cards"


def test_mode_difficulty_caps():
    st = {"p": 0.8, "attempts": 4, "correct": 3}
    assert m.target_difficulty(st, 1) == 2  # Beginner capped at 2 until mastered
    assert m.target_difficulty(st, 5) == 3


def test_leitner_scheduling():
    now = 1_000_000.0
    st = {"p": 0.7, "box": 1}
    st = m.update_state(st, MCQ, True, now=now)
    assert st["box"] == 2 and st["due_at"] == now + m.BOX_INTERVALS[2]
    st = m.update_state(st, MCQ, False, now=now)
    assert st["box"] == 1 and st["due_at"] == now + m.RETRY_AFTER_MISS


def test_unlock_rule():
    c = {"prereqs": ["a", "b"]}
    assert not m.is_unlocked(c, {"a": {"p": 0.9}, "b": {"p": 0.5}})
    assert m.is_unlocked(c, {"a": {"p": 0.9}, "b": {"p": 0.6}})
    assert m.is_unlocked(c, {}, free_play=True)


def test_mastery_reachable_in_reasonable_number_of_answers():
    st = {}
    for i in range(8):
        st = m.update_state(st, dict(MCQ, difficulty=1 + min(i // 2, 2)), True)
    assert m.status(st, True) == "mastered"
