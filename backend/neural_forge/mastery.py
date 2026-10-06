"""Adaptive learning engine: Bayesian Knowledge Tracing + Leitner spaced review + explicit adaptation rules.

See docs/ADAPTIVE_LEARNING.md for the full description. Summary:

1. **Mastery estimate (BKT).** For every concept we keep P(known). After each piece of evidence:
       correct:   P' = P(1-s) / (P(1-s) + (1-P)g)
       incorrect: P' = P s / (P s + (1-P)(1-g))
       then learning:  P'' = P' + (1-P')·T
   g (guess) depends on the question format (MCQ = 1/#options + 0.10, numeric ≈ 0.05), s (slip) on difficulty.
   Hints make a correct answer weaker evidence (g is raised); harder questions make it stronger.
2. **Unlocking.** A concept unlocks when every prerequisite has P(known) ≥ UNLOCK.
3. **Difficulty targeting.** P < 0.4 → easy items, P < 0.75 → medium, else hard (shifted by game mode).
4. **Struggle → remediation.** Two consecutive misses (or ≥4 attempts with P < 0.35) mark the concept as
   *struggling*: next time the game shows a mini-lesson + visual, serves an easier item, and schedules a retry
   of the original difficulty after the learner recovers.
5. **Mastery → acceleration.** P ≥ 0.85: teach cards are skipped ("quick check"), hard items only.
6. **Spaced review (Leitner).** Correct answers when proficient move the concept up a box
   (intervals 1, 3, 7, 16, 35 days); a miss drops it to box 1 with a 10-minute retry.
"""
from __future__ import annotations

import time

P_INIT = 0.10
P_TRANSIT = 0.15
UNLOCK = 0.60
PROFICIENT = 0.60
MASTERED = 0.85
SLIP = {1: 0.10, 2: 0.12, 3: 0.15}
MIN_CORRECT_FOR_MASTERY = 3  # P(known) alone is not enough: need at least 3 correct answers
BOX_INTERVALS = [0, 1 * 86400, 3 * 86400, 7 * 86400, 16 * 86400, 35 * 86400]
RETRY_AFTER_MISS = 600

MODES = {
    1: dict(name="Beginner", hints="auto", hint_cost=0.0, max_difficulty_until_mastered=2, start_shift=0,
            show_recommendations=True, show_code=False, explain_before=True, boss_bonus=0.0,
            desc="Everything explained, recommended choices, hints shown automatically after a miss, no penalties."),
    2: dict(name="Guided", hints="on_request", hint_cost=0.0, max_difficulty_until_mastered=3, start_shift=0,
            show_recommendations=False, show_code=False, explain_before=True, boss_bonus=0.0,
            desc="You choose between reasonable options; hints on request, free."),
    3: dict(name="Practice", hints="on_request", hint_cost=0.5, max_difficulty_until_mastered=3, start_shift=0,
            show_recommendations=False, show_code=False, explain_before=False, boss_bonus=0.0,
            desc="Mostly independent; optional hints cost half the XP."),
    4: dict(name="Engineer", hints="on_request", hint_cost=0.5, max_difficulty_until_mastered=3, start_shift=1,
            show_recommendations=False, show_code=True, explain_before=False, boss_bonus=0.0,
            desc="Real Python: generated code shown everywhere, Code Dojo emphasised, harder items earlier."),
    5: dict(name="Research Challenge", hints="none", hint_cost=1.0, max_difficulty_until_mastered=3, start_shift=1,
            show_recommendations=False, show_code=True, explain_before=False, boss_bonus=0.03,
            desc="Open-ended challenges scored on metrics and engineering quality; no hints; stricter boss thresholds."),
}


def guess_prob(item: dict) -> float:
    if item.get("type") == "mcq":
        # 1/#options plus 0.10 for partial-knowledge elimination of obviously wrong options
        return min(0.5, 1.0 / max(2, len(item.get("options", [])) or 4) + 0.10)
    if item.get("type") == "numeric":
        return 0.05
    if item.get("type") == "prediction":
        return 1.0 / max(2, item.get("n_options", 3))
    if item.get("type") in ("experiment", "code"):
        return 0.10
    return 0.25


def bkt_update(p: float, correct: bool, *, guess: float, slip: float, transit: float = P_TRANSIT) -> float:
    if correct:
        post = p * (1 - slip) / (p * (1 - slip) + (1 - p) * guess)
    else:
        post = p * slip / (p * slip + (1 - p) * (1 - guess))
    return min(0.999, max(0.001, post + (1 - post) * transit))


def evidence_params(item: dict, hint_used: bool) -> tuple[float, float, float]:
    d = int(item.get("difficulty", 2))
    g = guess_prob(item)
    if d == 3:
        g *= 0.7  # a correct hard answer is stronger evidence
    s = SLIP.get(d, 0.1)
    t = P_TRANSIT
    if hint_used:
        g = g + (1 - g) * 0.5
        t *= 0.5
    if item.get("type") in ("experiment", "code"):
        s, t = 0.15, 0.2
    return g, s, t


def update_state(state: dict, item: dict, correct: bool, hint_used: bool = False, now: float | None = None) -> dict:
    """Return a new mastery state after one piece of evidence. Pure function (easy to test)."""
    now = now or time.time()
    st = dict(state)
    g, s, t = evidence_params(item, hint_used)
    before = st.get("p", P_INIT)
    st["p"] = bkt_update(before, correct, guess=g, slip=s, transit=t)
    st["attempts"] = st.get("attempts", 0) + 1
    st["correct"] = st.get("correct", 0) + (1 if correct else 0)
    st["streak_wrong"] = 0 if correct else st.get("streak_wrong", 0) + 1
    st["streak_right"] = st.get("streak_right", 0) + 1 if correct else 0
    st["hint_uses"] = st.get("hint_uses", 0) + (1 if hint_used else 0)
    st["last_seen"] = now
    # struggle / remediation bookkeeping
    if not correct:
        st["failed_difficulty"] = max(st.get("failed_difficulty") or 0, int(item.get("difficulty", 1)))
    struggling = st["streak_wrong"] >= 2 or (st["attempts"] >= 4 and st["p"] < 0.35)
    if struggling:
        st["struggling"] = True
    elif st.get("struggling") and correct and st["streak_right"] >= 2:
        st["struggling"] = False
        st["retry_pending"] = True  # recovered on easier items → retry the original difficulty next
    if st.get("retry_pending") and correct and int(item.get("difficulty", 1)) >= (st.get("failed_difficulty") or 1):
        st["retry_pending"] = False
        st["failed_difficulty"] = None
    # Leitner scheduling
    box = st.get("box", 0)
    if correct and st["p"] >= PROFICIENT:
        box = min(box + 1, len(BOX_INTERVALS) - 1)
        st["due_at"] = now + BOX_INTERVALS[box]
    elif not correct:
        box = 1
        st["due_at"] = now + RETRY_AFTER_MISS
    st["box"] = box
    st["delta"] = st["p"] - before
    return st


def status(state: dict | None, unlocked: bool) -> str:
    if not unlocked:
        return "locked"
    if not state or not state.get("attempts"):
        return "new"
    if state.get("struggling"):
        return "struggling"
    p = state.get("p", P_INIT)
    if p >= MASTERED and state.get("correct", 0) >= MIN_CORRECT_FOR_MASTERY:
        return "mastered"
    if p >= PROFICIENT:
        return "proficient"
    return "learning"


def target_difficulty(state: dict | None, mode: int) -> int:
    cfg = MODES.get(mode, MODES[2])
    p = (state or {}).get("p", P_INIT)
    if state and state.get("struggling"):
        return 1
    if state and state.get("retry_pending"):
        return int(state.get("failed_difficulty") or 2)
    d = 1 if p < 0.4 else 2 if p < 0.75 else 3
    d = min(3, d + cfg["start_shift"])
    if p < MASTERED:
        d = min(d, cfg["max_difficulty_until_mastered"])
    return d


def is_unlocked(concept: dict, states: dict[str, dict], free_play: bool = False) -> bool:
    if free_play:
        return True
    return all(states.get(p, {}).get("p", P_INIT) >= UNLOCK for p in concept["prereqs"])


def lesson_plan(state: dict | None, mode: int) -> dict:
    """Which cards to show for a concept lesson, adapted to mastery and mode."""
    p = (state or {}).get("p", P_INIT)
    if state and state.get("struggling"):
        return dict(kind="remediation", cards=["mini_lesson", "visual", "example", "practice_easy", "practice_easy", "reflect"],
                    message="Let's slow down. Here's a simpler explanation and a visual, then an easier question. We'll retry the harder one afterwards.")
    if p >= MASTERED and (state or {}).get("correct", 0) >= MIN_CORRECT_FOR_MASTERY:
        return dict(kind="quick_check", cards=["challenge", "challenge"], message="You've mastered this — quick check with hard questions only.")
    if p >= PROFICIENT:
        return dict(kind="practice", cards=["recap", "practice", "challenge", "reflect"], message="You know the basics. Practise and take on a challenge.")
    cards = ["teach", "visual", "example", "guided", "predict", "practice", "challenge", "reflect"]
    if mode >= 4:
        cards.insert(3, "code")
    return dict(kind="full", cards=cards, message=None)


def xp_for_answer(item: dict, correct: bool, hint_used: bool, mode: int) -> int:
    if not correct:
        return 0
    base = 10 * int(item.get("difficulty", 1))
    if hint_used:
        base = int(base * (1 - MODES[mode]["hint_cost"]))
    return base
