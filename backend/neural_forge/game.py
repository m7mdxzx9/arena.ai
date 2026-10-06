"""Game service: ties curriculum, adaptive engine, labs, missions, bosses and persistence together.

Every public function takes a DB and a player id and returns JSON-serialisable dicts. The FastAPI layer
(app.py) is a thin wrapper around this module, which keeps the game logic testable without HTTP.
"""
from __future__ import annotations

import functools
import random
import time

from . import agent, bosses, code_exercises, datalab, datasets, mastery, missions, ml, nn, predictions, rag, sandbox
from .curriculum import AREAS, BRANCH_ORDER, CONCEPT_LIST, CONCEPTS, EQUIPMENT, RANKS, depth
from .curriculum import generators as gen
from .db import DB


class GameError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message, self.status = message, status


# ------------------------------------------------------------------ achievements
ACHIEVEMENTS = {
    "first_steps": ("First Steps", "Answer your first question correctly.", "👣"),
    "streak_5": ("On Fire", "5 correct answers in a row.", "🔥"),
    "first_run": ("Lab Coat", "Run your first real experiment.", "🧪"),
    "comparer": ("Fair Judge", "Compare two or more experiment runs.", "⚖️"),
    "data_detective": ("Data Detective", "Profile 5 different datasets in the Data Lab.", "🔍"),
    "predictor": ("Oracle", "Predict 5 experiment outcomes correctly.", "🔮"),
    "surprised": ("Humbled by Data", "Make a wrong prediction — and learn from it.", "😮"),
    "code_first": ("Hello, World", "Pass your first Code Dojo exercise.", "⌨️"),
    "coder_10": ("Engineer", "Pass 10 Code Dojo exercises.", "🛠️"),
    "reproducible": ("Reproducible", "Add notes to a saved experiment.", "📓"),
    "mastery_10": ("Rising Scholar", "Reach proficiency in 10 concepts.", "📘"),
    "mastery_50": ("Polymath", "Reach proficiency in 50 concepts.", "📚"),
    "area_complete": ("Building Restored", "Bring a campus building to full power (level 3).", "🏗️"),
    "reflective": ("Reflective Practitioner", "Write 5 reflections.", "🪞"),
    "mission_first": ("Field Agent", "Complete your first mission.", "🎯"),
    "researcher": ("Published", "Submit a Research Challenge entry.", "🔭"),
    "all_bosses": ("Campus Guardian", "Defeat all 8 bosses.", "🛡️"),
}
for _b in bosses.BOSSES:
    ACHIEVEMENTS[f"boss_{_b['id']}"] = (f"Defeated {_b['name'].title()}", _b["taunt"][:60] + "…", _b["icon"])


def _grant(db: DB, pid: int, ach: str, out: list):
    if db.grant(pid, ach):
        name, desc, icon = ACHIEVEMENTS[ach]
        out.append(dict(id=ach, name=name, desc=desc, icon=icon))


# ------------------------------------------------------------------ helpers
def _player(db: DB, pid: int) -> dict:
    p = db.player(pid)
    if not p:
        raise GameError("Player not found", 404)
    return p


def _states(db: DB, pid: int) -> dict[str, dict]:
    return db.mastery(pid)


def _unlocked(c: dict, states: dict, p: dict) -> bool:
    return mastery.is_unlocked(c, states, p["settings"].get("free_play", False))


def concept_status(c: dict, states: dict, p: dict) -> str:
    return mastery.status(states.get(c["id"]), _unlocked(c, states, p))


def rank_info(xp: int, n_mastered: int, n_bosses: int) -> dict:
    idx = 0
    for i, (_, need_xp, need_m, need_b) in enumerate(RANKS):
        if xp >= need_xp and n_mastered >= need_m and n_bosses >= need_b:
            idx = i
    nxt = RANKS[idx + 1] if idx + 1 < len(RANKS) else None
    return dict(index=idx, title=RANKS[idx][0], all=[r[0] for r in RANKS],
                next=dict(title=nxt[0], xp=nxt[1], proficient=nxt[2], bosses=nxt[3]) if nxt else None)


def area_levels(states: dict, p: dict, defeated: set[str]) -> dict:
    out = {}
    for aid, a in AREAS.items():
        cs = [c for c in CONCEPT_LIST if c["area"] == aid]
        prof = sum(1 for c in cs if states.get(c["id"], {}).get("p", 0) >= mastery.PROFICIENT)
        frac = prof / len(cs) if cs else 0
        level = 3 if frac >= 0.999 else 2 if frac >= 0.5 else 1 if frac >= 0.15 else 0
        area_bosses = [b["id"] for b in bosses.BOSSES if b["area"] == aid]
        unlocked = any(_unlocked(c, states, p) for c in cs)
        out[aid] = dict(level=level, proficient=prof, total=len(cs), fraction=round(frac, 3), unlocked=unlocked,
                        bosses=area_bosses, bosses_defeated=[b for b in area_bosses if b in defeated])
    return out


def _award_mastery_transitions(db, pid, cid, before: dict, after: dict, events: list) -> int:
    xp = 0
    if after.get("p", 0) >= mastery.PROFICIENT and not before.get("awarded_prof"):
        after["awarded_prof"] = True
        xp += 25
        events.append(dict(kind="proficient", concept=cid, text=f"Proficient in {CONCEPTS[cid]['name']}! +25 XP"))
    if mastery.status(after, True) == "mastered" and not before.get("awarded_mast"):
        after["awarded_mast"] = True
        xp += 50
        events.append(dict(kind="mastered", concept=cid, text=f"Mastered {CONCEPTS[cid]['name']}! +50 XP"))
    return xp


def _newly_unlocked(states_before: dict, states_after: dict, p: dict) -> list[dict]:
    out = []
    for c in CONCEPT_LIST:
        if not _unlocked(c, states_before, p) and _unlocked(c, states_after, p):
            out.append(dict(id=c["id"], name=c["name"], area=c["area"]))
    return out


def evidence(db: DB, pid: int, cid: str, item: dict, correct: bool, hint: bool = False, kind: str = "experiment", key: str = "") -> dict:
    """Apply one piece of evidence to a concept; handles XP for transitions and unlock detection."""
    p = _player(db, pid)
    states = _states(db, pid)
    before = states.get(cid, {})
    after = mastery.update_state(before, item, correct, hint)
    for k in ("awarded_prof", "awarded_mast"):
        if before.get(k):
            after[k] = True
    events: list = []
    xp = _award_mastery_transitions(db, pid, cid, before, after, events)
    db.set_mastery(pid, cid, after)
    db.log_attempt(pid, cid, kind, key, int(item.get("difficulty", 2)), correct, hint, before.get("p", mastery.P_INIT), after["p"])
    new_states = dict(states, **{cid: after})
    unlocked = _newly_unlocked(states, new_states, p)
    if xp:
        db.add_xp(pid, xp, f"{CONCEPTS[cid]['name']}: milestone")
    return dict(state=after, events=events, unlocked=unlocked, xp=xp, status=mastery.status(after, True))


def _check_counts(db: DB, pid: int, out: list):
    states = _states(db, pid)
    prof = sum(1 for s in states.values() if s.get("p", 0) >= mastery.PROFICIENT)
    if prof >= 10:
        _grant(db, pid, "mastery_10", out)
    if prof >= 50:
        _grant(db, pid, "mastery_50", out)
    p = _player(db, pid)
    defeated = {k for k, v in db.progress(pid, "boss").items() if v.get("completed_at")}
    if any(a["level"] == 3 for a in area_levels(states, p, defeated).values()):
        _grant(db, pid, "area_complete", out)


# ------------------------------------------------------------------ players & overview
def create_player(db: DB, name: str, mode: int) -> dict:
    name = (name or "").strip()[:40] or "Apprentice"
    if mode not in mastery.MODES:
        raise GameError("Unknown difficulty mode")
    pid = db.create_player(name, mode)
    return overview(db, pid)


def set_settings(db: DB, pid: int, mode: int | None = None, free_play: bool | None = None, show_code: bool | None = None) -> dict:
    p = _player(db, pid)
    s = dict(p["settings"])
    fields = {}
    if mode is not None:
        if mode not in mastery.MODES:
            raise GameError("Unknown difficulty mode")
        fields["mode"] = mode
        if show_code is None:
            s["show_code"] = mode >= 4
    if free_play is not None:
        s["free_play"] = bool(free_play)
    if show_code is not None:
        s["show_code"] = bool(show_code)
    fields["settings"] = s
    db.update_player(pid, **fields)
    return overview(db, pid)


def recommendations(db: DB, pid: int, limit: int = 6) -> list[dict]:
    p = _player(db, pid)
    states = _states(db, pid)
    now = time.time()
    recs = []
    due = [(s.get("due_at", 0), cid) for cid, s in states.items() if s.get("box", 0) >= 2 and s.get("due_at", 1e18) <= now]
    for _, cid in sorted(due)[:2]:
        recs.append(dict(kind="review", id=cid, title=f"Review: {CONCEPTS[cid]['name']}", why="Spaced review is due — revisiting just before you forget locks it in."))
    for cid, s in states.items():
        if s.get("struggling") or s.get("retry_pending"):
            recs.append(dict(kind="lesson", id=cid, title=f"Retry: {CONCEPTS[cid]['name']}",
                             why="You found this tricky. A simpler explanation and an easier question are ready."))
    learnable = [c for c in CONCEPT_LIST if _unlocked(c, states, p) and states.get(c["id"], {}).get("p", 0) < mastery.PROFICIENT
                 and not states.get(c["id"], {}).get("struggling")]
    learnable.sort(key=lambda c: (depth(c["id"]), BRANCH_ORDER.index(c["branch"])))
    for c in learnable[:3]:
        started = c["id"] in states
        recs.append(dict(kind="lesson", id=c["id"], title=("Continue: " if started else "Learn: ") + c["name"],
                         why="All prerequisites are in place." if c["prereqs"] else "A starting point — no prerequisites."))
    mprog = db.progress(pid, "mission")
    for m in missions.MISSIONS:
        if not mprog.get(m["id"], {}).get("completed_at") and all(states.get(r, {}).get("p", 0) >= mastery.UNLOCK for r in m["requires"]):
            recs.append(dict(kind="mission", id=m["id"], title=f"Mission: {m['title']}", why="You have the skills for this real-data mission."))
            break
    bprog = db.progress(pid, "boss")
    for b in bosses.BOSSES:
        if not bprog.get(b["id"], {}).get("completed_at") and all(states.get(r, {}).get("p", 0) >= mastery.UNLOCK for r in b["requires"]):
            recs.append(dict(kind="boss", id=b["id"], title=f"Boss: {b['name']}", why="A boss stirs. You're ready."))
            break
    seen, out = set(), []
    for r in recs:
        if (r["kind"], r["id"]) not in seen:
            seen.add((r["kind"], r["id"])); out.append(r)
    return out[:limit]


def overview(db: DB, pid: int) -> dict:
    p = _player(db, pid)
    states = _states(db, pid)
    statuses = {c["id"]: concept_status(c, states, p) for c in CONCEPT_LIST}
    prof = sum(1 for s in statuses.values() if s in ("proficient", "mastered"))
    mast = sum(1 for s in statuses.values() if s == "mastered")
    defeated = {k for k, v in db.progress(pid, "boss").items() if v.get("completed_at")}
    equipment = {k: dict(**v, unlocked=all(states.get(r, {}).get("p", 0) >= mastery.UNLOCK for r in v["requires"]) or p["settings"].get("free_play"))
                 for k, v in EQUIPMENT.items()}
    ach = db.achievements(pid)
    return dict(player=dict(id=p["id"], name=p["name"], mode=p["mode"], xp=p["xp"], settings=p["settings"]),
                mode=dict(id=p["mode"], **mastery.MODES[p["mode"]]), modes={k: dict(name=v["name"], desc=v["desc"]) for k, v in mastery.MODES.items()},
                rank=rank_info(p["xp"], prof, len(defeated)),
                counts=dict(concepts=len(CONCEPT_LIST), proficient=prof, mastered=mast,
                            learning=sum(1 for s in statuses.values() if s in ("learning", "struggling")),
                            missions_done=sum(1 for v in db.progress(pid, "mission").values() if v.get("completed_at")), missions=len(missions.MISSIONS),
                            bosses_defeated=len(defeated), bosses=len(bosses.BOSSES),
                            exercises_done=sum(1 for v in db.progress(pid, "exercise").values() if v.get("completed_at")), exercises=len(code_exercises.EXERCISES)),
                areas={aid: dict(**AREAS[aid], **lv) for aid, lv in area_levels(states, p, defeated).items()},
                equipment=equipment, recommendations=recommendations(db, pid),
                achievements=[dict(id=k, name=v[0], desc=v[1], icon=v[2], earned_at=ach.get(k)) for k, v in ACHIEVEMENTS.items()],
                predictions={k: dict(choice=v.get("choice"), was_right=v.get("right")) for k, v in db.progress(pid, "prediction").items()},
                events=db.events(pid, 12))


def tree(db: DB, pid: int) -> dict:
    p = _player(db, pid)
    states = _states(db, pid)
    nodes = []
    for c in CONCEPT_LIST:
        s = states.get(c["id"], {})
        nodes.append(dict(id=c["id"], name=c["name"], branch=c["branch"], area=c["area"], prereqs=c["prereqs"], depth=depth(c["id"]),
                          status=concept_status(c, states, p), p=round(s.get("p", mastery.P_INIT), 3), attempts=s.get("attempts", 0),
                          due=bool(s.get("box", 0) >= 2 and s.get("due_at", 1e18) <= time.time())))
    return dict(branches=BRANCH_ORDER, nodes=nodes, thresholds=dict(unlock=mastery.UNLOCK, proficient=mastery.PROFICIENT, mastered=mastery.MASTERED))


# ------------------------------------------------------------------ lessons & questions
def _linked(cid: str) -> dict:
    return dict(exercises=[e["id"] for e in code_exercises.EXERCISES if e["concept"] == cid],
                missions=[m["id"] for m in missions.MISSIONS if cid in m["concepts"]],
                bosses=[b["id"] for b in bosses.BOSSES if cid in b["requires"]])


def lesson(db: DB, pid: int, cid: str) -> dict:
    p = _player(db, pid)
    c = CONCEPTS.get(cid)
    if not c:
        raise GameError("Unknown concept", 404)
    states = _states(db, pid)
    if not _unlocked(c, states, p):
        missing = [dict(id=r, name=CONCEPTS[r]["name"], p=round(states.get(r, {}).get("p", mastery.P_INIT), 2))
                   for r in c["prereqs"] if states.get(r, {}).get("p", mastery.P_INIT) < mastery.UNLOCK]
        raise GameError(f"Locked. Reach {int(mastery.UNLOCK * 100)}% mastery in: " + ", ".join(m["name"] for m in missing), 403)
    st = states.get(cid)
    plan = mastery.lesson_plan(st, p["mode"])
    cards = list(plan["cards"])
    if "predict" in cards and not c["predict"]:
        cards.remove("predict")
    if "visual" in cards and not c["visual"]:
        cards.remove("visual")
    if "example" in cards and not c["example"]:
        cards.remove("example")
    linked = _linked(cid)
    if "code" in cards and not (c["code"] or linked["exercises"]):
        cards.remove("code")
    if not c["reflect"] and "reflect" in cards:
        cards.remove("reflect")
    pred = None
    if c["predict"]:
        e = predictions.EXPERIMENTS[c["predict"]]
        pred = dict(id=c["predict"], title=e["title"], setup=e["setup"], question=e["question"], options=e["options"])
    area = AREAS[c["area"]]
    return dict(concept=dict(id=c["id"], name=c["name"], branch=c["branch"], area=c["area"], explain=c["explain"], analogy=c["analogy"],
                             visual=c["visual"], example=c["example"], reflect=c["reflect"], code=c["code"], prereqs=c["prereqs"]),
                mentor=dict(name=area["mentor"], area=area["name"], icon=area["icon"], color=area["color"]),
                plan=dict(kind=plan["kind"], message=plan["message"], cards=cards),
                state=dict(p=round((st or {}).get("p", mastery.P_INIT), 3), status=mastery.status(st, True), attempts=(st or {}).get("attempts", 0)),
                target_difficulty=mastery.target_difficulty(st, p["mode"]), prediction=pred, linked=linked,
                mode=dict(id=p["mode"], hints=mastery.MODES[p["mode"]]["hints"], show_code=p["settings"].get("show_code", False)))


def _item_from_key(cid: str, key: str) -> dict:
    c = CONCEPTS[cid]
    kind, *rest = key.split(":")
    if kind == "s":
        return c["questions"][int(rest[0])]
    if kind == "g":
        name, d, seed = rest[0], int(rest[1]), int(rest[2])
        if name not in c["generators"]:
            raise GameError("Bad item")
        return gen.generate(name, d, seed)
    raise GameError("Bad item")


def next_question(db: DB, pid: int, cid: str, purpose: str = "practice") -> dict:
    p = _player(db, pid)
    c = CONCEPTS.get(cid)
    if not c:
        raise GameError("Unknown concept", 404)
    st = _states(db, pid).get(cid)
    target = mastery.target_difficulty(st, p["mode"])
    cap = mastery.MODES[p["mode"]]["max_difficulty_until_mastered"] if mastery.status(st, True) != "mastered" else 3
    if purpose in ("guided", "practice_easy"):
        target = 1
    elif purpose == "challenge":
        target = min(cap, target + 1, 3)
    seen = db.seen_items(pid, cid)
    qs = c["questions"]
    candidates = [(f"s:{i}", q) for i, q in enumerate(qs) if q["difficulty"] == target and f"s:{i}" not in seen[:12]]
    unseen = [x for x in candidates if x[0] not in seen]
    pool = unseen or ([] if c["generators"] else candidates)
    if pool:
        key, q = random.choice(pool)
    elif c["generators"]:
        name = random.choice(c["generators"])
        seed = random.randrange(10 ** 6)
        key, q = f"g:{name}:{target}:{seed}", gen.generate(name, target, seed)
    else:
        near = sorted(range(len(qs)), key=lambda i: (abs(qs[i]["difficulty"] - target), f"s:{i}" in seen[:3], random.random()))
        key, q = f"s:{near[0]}", qs[near[0]]
    mode = mastery.MODES[p["mode"]]
    out = dict(key=key, type=q["type"], difficulty=q["difficulty"], prompt=q["prompt"], purpose=purpose,
               hints=mode["hints"], hint_available=mode["hints"] != "none", hint_cost=mode["hint_cost"])
    if q["type"] == "mcq":
        out["options"] = q["options"]
    if purpose == "guided" and mode["hints"] != "none":
        out["hint"] = _hint_text(c, q)  # guided practice: hint is part of the scaffold, no penalty
    return out


def _hint_text(c: dict, q: dict) -> str:
    if q.get("hint"):
        return q["hint"]
    if c["analogy"]:
        return "Think of it this way: " + c["analogy"].split(". ")[0].rstrip(".") + "."
    return "Re-read the key idea: " + c["explain"].split(". ")[0].rstrip(".") + "."


def hint(db: DB, pid: int, cid: str, key: str) -> dict:
    p = _player(db, pid)
    if mastery.MODES[p["mode"]]["hints"] == "none":
        raise GameError("Hints are disabled in Research Challenge mode.")
    return dict(hint=_hint_text(CONCEPTS[cid], _item_from_key(cid, key)), cost=mastery.MODES[p["mode"]]["hint_cost"])


def _grade(q: dict, response) -> bool:
    try:
        if q["type"] == "mcq":
            return int(response) == q["answer"]
        return abs(float(str(response).replace(",", ".")) - float(q["answer"])) <= q["tolerance"] + 1e-9
    except (TypeError, ValueError):
        return False


def answer(db: DB, pid: int, cid: str, key: str, response, hint_used: bool = False, purpose: str = "practice") -> dict:
    p = _player(db, pid)
    c = CONCEPTS.get(cid)
    if not c:
        raise GameError("Unknown concept", 404)
    q = _item_from_key(cid, key)
    correct = _grade(q, response)
    scaffolded = purpose == "guided"
    ev = evidence(db, pid, cid, q, correct, hint_used and not scaffolded, kind="question", key=key)
    xp = mastery.xp_for_answer(q, correct, hint_used and not scaffolded, p["mode"])
    settings = dict(p["settings"])
    settings["streak"] = settings.get("streak", 0) + 1 if correct else 0
    bonus = 5 if correct and settings["streak"] % 3 == 0 else 0
    db.update_player(pid, settings=settings)
    if xp + bonus:
        db.add_xp(pid, xp + bonus, f"{c['name']}: correct answer" + (" (+streak)" if bonus else ""))
    new_ach: list = []
    if correct:
        _grant(db, pid, "first_steps", new_ach)
    if settings["streak"] >= 5:
        _grant(db, pid, "streak_5", new_ach)
    _check_counts(db, pid, new_ach)
    st = ev["state"]
    adapt = None
    if st.get("struggling"):
        adapt = "This concept seems tricky right now — next, a simpler explanation, a visual and an easier question. We'll come back to the harder one."
    elif st.get("retry_pending"):
        adapt = "Nice recovery! Next you'll retry a question at the level you missed earlier."
    elif mastery.status(st, True) == "mastered":
        adapt = "Mastered — from now on you'll get hard questions only, and this concept enters spaced review."
    auto_hint = (not correct and mastery.MODES[p["mode"]]["hints"] == "auto")
    return dict(correct=correct, explanation=q["explanation"], answer=q["answer"],
                answer_text=q["options"][q["answer"]] if q["type"] == "mcq" else str(q["answer"]),
                hint=_hint_text(c, q) if auto_hint else None,
                xp=xp + bonus + ev["xp"], streak=settings["streak"], p_before=round(st["p"] - st["delta"], 3), p=round(st["p"], 3),
                status=ev["status"], struggling=bool(st.get("struggling")), adaptation=adapt, events=ev["events"], unlocked=ev["unlocked"],
                achievements=new_ach, next_difficulty=mastery.target_difficulty(st, p["mode"]))


def reflect(db: DB, pid: int, cid: str, text: str) -> dict:
    text = (text or "").strip()
    if len(text) < 10:
        raise GameError("Write at least a sentence — reflection is where learning sticks.")
    db.add_reflection(pid, cid, text)
    db.add_xp(pid, 10, f"Reflection: {CONCEPTS.get(cid, {}).get('name', cid)}")
    out: list = []
    if len(db.reflections(pid)) >= 5:
        _grant(db, pid, "reflective", out)
    return dict(ok=True, xp=10, achievements=out)


def review_queue(db: DB, pid: int) -> list[dict]:
    now = time.time()
    st = _states(db, pid)
    return [dict(id=cid, name=CONCEPTS[cid]["name"], box=s.get("box", 0), due_at=s.get("due_at"), p=round(s["p"], 3))
            for cid, s in sorted(st.items(), key=lambda kv: kv[1].get("due_at", 0))
            if cid in CONCEPTS and s.get("box", 0) >= 1 and s.get("due_at", 1e18) <= now]


# ------------------------------------------------------------------ predictions
def prediction(db: DB, pid: int, exp_id: str, choice: int) -> dict:
    if exp_id not in predictions.EXPERIMENTS:
        raise GameError("Unknown experiment", 404)
    res = predictions.run(exp_id, int(choice))
    prog = db.progress(pid, "prediction")
    first = exp_id not in prog
    xp = (20 if res["was_right"] else 5) if first else 0
    if xp:
        db.add_xp(pid, xp, f"Prediction: {predictions.EXPERIMENTS[exp_id]['title']}")
    db.set_progress(pid, "prediction", exp_id, dict(choice=int(choice), right=res["was_right"]), completed=True)
    item = dict(type="prediction", n_options=len(res["options"]), difficulty=2)
    ev = evidence(db, pid, res["concept"], item, res["was_right"], kind="prediction", key=exp_id) if first else None
    out: list = []
    prog = db.progress(pid, "prediction")
    if sum(1 for v in prog.values() if v.get("right")) >= 5:
        _grant(db, pid, "predictor", out)
    if not res["was_right"]:
        _grant(db, pid, "surprised", out)
    return dict(**res, xp=xp, first=first, achievements=out, mastery=ev and dict(p=round(ev["state"]["p"], 3), status=ev["status"]))


# ------------------------------------------------------------------ labs (runs are saved as experiment history)
def _summary_ml(cfg, res):
    m = res["metrics"]
    return dict(dataset=cfg["dataset"], model=cfg["model"], model_label=ml.MODELS[cfg["model"]]["label"], task=res["task"],
                primary=res.get("primary_metric"), test=m.get("test"), train=m.get("train"), cv=res.get("cv"),
                n_features=len(res.get("features_used", [])))


def run_ml(db: DB, pid: int, cfg: dict, context: str | None = None, name: str | None = None) -> dict:
    if cfg.get("dataset") not in datasets.REGISTRY:
        raise GameError("Unknown dataset")
    if cfg.get("model") not in ml.MODELS:
        raise GameError("Unknown model")
    exp = ml.Experiment.from_dict(cfg)
    try:
        res = ml.run_experiment(exp)
    except ml.PipelineError as e:
        return dict(ok=False, **e.to_dict())
    full_cfg = dict(dataset=exp.dataset, target=exp.target, features=exp.features, model=exp.model, params=exp.params,
                    preprocessing={**ml.DEFAULT_PREP, **(exp.preprocessing or {})}, test_size=exp.test_size, seed=exp.seed,
                    cv_folds=exp.cv_folds, threshold=exp.threshold, extra_rows=exp.extra_rows)
    rid = db.save_run(pid, "ml", full_cfg, res, _summary_ml(full_cfg, res), name=name, context=context)
    return dict(ok=True, run_id=rid, config=full_cfg, result=res, **_after_run(db, pid))


def _after_run(db, pid) -> dict:
    out: list = []
    _grant(db, pid, "first_run", out)
    if out:
        db.add_xp(pid, 20, "First experiment")
    return dict(achievements=out)


def run_nn(db: DB, pid: int, cfg: dict, context: str | None = None, name: str | None = None) -> dict:
    try:
        res = nn.train(cfg)
    except (ValueError, KeyError) as e:
        return dict(ok=False, error=str(e), lesson="Check the configuration.", fix="")
    summary = dict(dataset=cfg.get("dataset"), hidden=res["sizes"][1:-1], final=res["final"], diverged_at=res["diverged_at"],
                   diagnosis=[d["code"] for d in res["diagnosis"]])
    rid = db.save_run(pid, "nn", cfg, res, summary, name=name, context=context)
    return dict(ok=True, run_id=rid, config=cfg, result=res, **_after_run(db, pid))


def run_rag(db: DB, pid: int, cfg: dict, context: str | None = None, name: str | None = None) -> dict:
    res = rag.evaluate_retrieval(cfg)
    rid = db.save_run(pid, "rag", cfg, res, dict(metrics=res["metrics"]), name=name, context=context)
    return dict(ok=True, run_id=rid, config=cfg, result=res, **_after_run(db, pid))


def run_answers(db: DB, pid: int, cfg: dict, context: str | None = None, name: str | None = None) -> dict:
    res = rag.evaluate_answers(cfg)
    rid = db.save_run(pid, "answers", cfg, res, dict(metrics=res["metrics"]), name=name, context=context)
    return dict(ok=True, run_id=rid, config=cfg, result=res, **_after_run(db, pid))


def run_agent(db: DB, pid: int, cfg: dict, context: str | None = None, name: str | None = None) -> dict:
    res = agent.evaluate(list(cfg.get("defences", [])))
    rid = db.save_run(pid, "agent", dict(defences=res["defences"]), res, dict(metrics=res["metrics"]), name=name, context=context)
    return dict(ok=True, run_id=rid, config=dict(defences=res["defences"]), result=res, **_after_run(db, pid))


RUNNERS = dict(ml=run_ml, nn=run_nn, rag=run_rag, answers=run_answers, agent=run_agent)


def compare(db: DB, pid: int, run_ids: list[int]) -> dict:
    runs = [db.run(pid, int(r)) for r in run_ids]
    runs = [r for r in runs if r]
    if len(runs) < 2:
        raise GameError("Pick at least two runs to compare.")
    kinds = {r["kind"] for r in runs}
    if kinds != {"ml"}:
        raise GameError("Comparison currently supports ML Workbench runs.")
    tasks = {r["result"]["task"] for r in runs}
    if len(tasks) > 1:
        raise GameError("Compare runs of the same task type (e.g. all classification).")
    out: list = []
    _grant(db, pid, "comparer", out)
    res = ml.compare_runs(runs)
    return dict(**res, achievements=out)


def profile(db: DB, pid: int, ds_id: str) -> dict:
    if ds_id not in datasets.REGISTRY:
        raise GameError("Unknown dataset", 404)
    prog = db.progress(pid, "profiled")
    db.set_progress(pid, "profiled", ds_id, {}, completed=True)
    out: list = []
    if len(prog) + (ds_id not in prog) >= 5:
        _grant(db, pid, "data_detective", out)
    return dict(**_profile_cached(ds_id), achievements=out)


@functools.lru_cache(maxsize=64)
def _profile_cached(ds_id):
    reg = datasets.REGISTRY[ds_id]
    df = datasets.load(ds_id)
    return dict(dataset=dict(id=ds_id, title=reg["title"], task=reg["task"], target=reg["target"], desc=reg.get("desc", ""), story=reg.get("story", "")),
                profile=datalab.profile(df, reg["target"]))


# ------------------------------------------------------------------ missions
def _bonus(p):
    return mastery.MODES[p["mode"]]["boss_bonus"]


def missions_list(db: DB, pid: int) -> list[dict]:
    p = _player(db, pid)
    states = _states(db, pid)
    prog = db.progress(pid, "mission")
    out = []
    for m in missions.MISSIONS:
        unlocked = p["settings"].get("free_play") or all(states.get(r, {}).get("p", 0) >= mastery.UNLOCK for r in m["requires"])
        out.append(dict(id=m["id"], title=m["title"], area=m["area"], dataset=m["dataset"], xp=m["xp"], requires=[dict(id=r, name=CONCEPTS[r]["name"]) for r in m["requires"]],
                        unlocked=bool(unlocked), completed=bool(prog.get(m["id"], {}).get("completed_at")), step=prog.get(m["id"], {}).get("step", 0)))
    return out


@functools.lru_cache(maxsize=32)
def _data_q(mid):
    return missions.data_question(missions.BY_ID[mid])


def mission(db: DB, pid: int, mid: str) -> dict:
    p = _player(db, pid)
    m = missions.BY_ID.get(mid)
    if not m:
        raise GameError("Unknown mission", 404)
    dq = _data_q(mid)
    st = db.progress(pid, "mission").get(mid, {})
    mode = mastery.MODES[p["mode"]]
    refl = m["reflection"]
    return dict(id=mid, title=m["title"], area=m["area"], mentor=AREAS[m["area"]]["mentor"], dataset=m["dataset"], task=m["task"],
                target=datasets.REGISTRY[m["dataset"]]["target"], concepts=[dict(id=c, name=CONCEPTS[c]["name"]) for c in m["concepts"]],
                briefing=m["briefing"], objectives=[o["label"] for o in m["objectives"]], banned=m["banned"], xp=m["xp"],
                data_question=dict(prompt=dq["prompt"], options=dq["options"]), metric_buckets=m["metric_buckets"],
                reflection=dict(prompt=refl["prompt"], options=refl["options"]) if refl else None,
                recommended=m["recommended"] if mode["show_recommendations"] else None, mentor_tip=m["mentor_tip"],
                bonus=_bonus(p), state=st)


def mission_action(db: DB, pid: int, mid: str, action: str, payload: dict) -> dict:
    p = _player(db, pid)
    m = missions.BY_ID.get(mid)
    if not m:
        raise GameError("Unknown mission", 404)
    prog = db.progress(pid, "mission").get(mid, {})
    st = {k: v for k, v in prog.items() if k != "completed_at"}
    st.setdefault("step", 0)
    out: dict = dict(ok=True)
    if action == "data_answer":
        dq = _data_q(mid)
        ok = int(payload.get("answer", -1)) == dq["answer"]
        st["data_correct"] = ok
        st["step"] = max(st["step"], 1)
        out.update(correct=ok, explanation=dq["explanation"], answer=dq["answer"])
    elif action == "hypothesis":
        st["hypothesis"] = str(payload.get("text", ""))[:500]
        st["prediction"] = payload.get("bucket")
        st["step"] = max(st["step"], 2)
    elif action == "submit_run":
        run = db.run(pid, int(payload.get("run_id", 0)))
        if not run or run["kind"] != "ml":
            raise GameError("Run not found")
        chk = missions.check_objectives(m, run, _bonus(p))
        actual = missions.bucket_index(m, run)
        pred = st.get("prediction")
        st["attempts"] = st.get("attempts", 0) + 1
        st["last_run"] = run["id"]
        out.update(check=chk, actual_bucket=actual, predicted_bucket=pred,
                   prediction_right=(pred is not None and actual is not None and int(pred) == actual))
        if chk["passed"]:
            st["passed_run"] = run["id"]
            st["step"] = max(st["step"], 3)
        if chk["passed"] and not st.get("evidence_given"):
            st["evidence_given"] = True
            for cid in m["concepts"]:
                evidence(db, pid, cid, dict(type="experiment", difficulty=2), True, kind="mission", key=mid)
    elif action == "reflect":
        if not st.get("passed_run"):
            raise GameError("Complete the objectives first.")
        refl = m["reflection"]
        ok = int(payload.get("answer", -1)) == refl["answer"]
        out.update(correct=ok, explanation=refl["explanation"], answer=refl["answer"])
        if not prog.get("completed_at"):
            xp = m["xp"] + (25 if st.get("data_correct") else 0) + (25 if ok else 0)
            db.add_xp(pid, xp, f"Mission complete: {m['title']}")
            out["xp"] = xp
            ach: list = []
            _grant(db, pid, "mission_first", ach)
            _check_counts(db, pid, ach)
            out["achievements"] = ach
        st["step"] = 4
        db.set_progress(pid, "mission", mid, st, completed=True)
        out["state"] = st
        out["completed"] = True
        return out
    else:
        raise GameError("Unknown action")
    db.set_progress(pid, "mission", mid, st)
    out["state"] = st
    return out


# ------------------------------------------------------------------ research challenges
def challenges_list(db: DB, pid: int) -> list[dict]:
    prog = db.progress(pid, "challenge")
    return [dict(**c, best=prog.get(c["id"], {}).get("best"), entries=prog.get(c["id"], {}).get("entries", [])) for c in missions.CHALLENGES]


def challenge_submit(db: DB, pid: int, ch_id: str, run_id: int) -> dict:
    if ch_id not in {c["id"] for c in missions.CHALLENGES}:
        raise GameError("Unknown challenge", 404)
    run = db.run(pid, int(run_id))
    if not run or run["kind"] != "ml":
        raise GameError("Run not found")
    sc = missions.score_challenge(ch_id, run)
    if not sc["valid"]:
        return sc
    prog = db.progress(pid, "challenge").get(ch_id, {})
    entries = (prog.get("entries") or []) + [dict(run_id=run["id"], score=sc["score"], ts=time.time())]
    best = max(e["score"] for e in entries)
    improved = best > (prog.get("best") or -1e9)
    db.set_progress(pid, "challenge", ch_id, dict(best=best, entries=entries[-20:]), completed=True)
    xp = int(round(max(0, sc["score"]))) if improved else 0
    if xp:
        db.add_xp(pid, xp, f"Research challenge new best: {sc['score']}")
    ach: list = []
    _grant(db, pid, "researcher", ach)
    return dict(**sc, best=best, improved=improved, xp=xp, achievements=ach)


# ------------------------------------------------------------------ bosses
@functools.lru_cache(maxsize=16)
def _boss_intro(bid):
    intro = bosses.BY_ID[bid]["intro"]()
    if intro["kind"] == "profile":
        intro["data"] = _profile_cached(intro["dataset"])
    return intro


def bosses_list(db: DB, pid: int) -> list[dict]:
    p = _player(db, pid)
    states = _states(db, pid)
    prog = db.progress(pid, "boss")
    out = []
    for b in bosses.BOSSES:
        st = prog.get(b["id"], {})
        ready = p["settings"].get("free_play") or all(states.get(r, {}).get("p", 0) >= mastery.UNLOCK for r in b["requires"])
        out.append(dict(id=b["id"], name=b["name"], icon=b["icon"], area=b["area"], xp=b["xp"], taunt=b["taunt"],
                        requires=[dict(id=r, name=CONCEPTS[r]["name"], p=round(states.get(r, {}).get("p", mastery.P_INIT), 2)) for r in b["requires"]],
                        unlocked=bool(ready), defeated=bool(st.get("completed_at")), hp=st.get("hp", 100), phase=st.get("phase", 0)))
    return out


def boss(db: DB, pid: int, bid: str) -> dict:
    p = _player(db, pid)
    b = bosses.BY_ID.get(bid)
    if not b:
        raise GameError("Unknown boss", 404)
    st = db.progress(pid, "boss").get(bid, {})
    return dict(**bosses.public(b), intro=_boss_intro(bid), state=dict(phase=st.get("phase", 0), hp=st.get("hp", 100), mistakes=st.get("mistakes", 0),
                                                                         defeated=bool(st.get("completed_at")), log=st.get("log", [])),
                bonus=_bonus(p), mode=p["mode"])


def boss_action(db: DB, pid: int, bid: str, payload: dict) -> dict:
    p = _player(db, pid)
    b = bosses.BY_ID.get(bid)
    if not b:
        raise GameError("Unknown boss", 404)
    prog = db.progress(pid, "boss").get(bid, {})
    st = {k: v for k, v in prog.items() if k != "completed_at"}
    st.setdefault("phase", 0); st.setdefault("hp", 100); st.setdefault("mistakes", 0); st.setdefault("log", [])
    if payload.get("restart"):
        st = dict(phase=0, hp=100, mistakes=0, log=[])
        db.set_progress(pid, "boss", bid, st)
        return dict(ok=True, state=st)
    if st["phase"] >= len(b["phases"]):
        raise GameError("Already defeated. Restart to fight again.")
    ph = b["phases"][st["phase"]]
    result = config = None
    if ph["kind"] not in ("mcq", "select"):
        run = db.run(pid, int(payload.get("run_id", 0)))
        if not run or run["kind"] != ph["kind"]:
            raise GameError(f"Submit a {ph['kind']} run for this phase.")
        result, config = run["result"], run["config"]
    ev = bosses.evaluate_phase(b, st["phase"], payload, result, config, _bonus(p))
    dmg = 0
    out: dict = dict(ok=True, evaluation=ev)
    concept = b["requires"][0]
    if ev["passed"]:
        n = len(b["phases"])
        dmg = st["hp"] - round(100 * (n - st["phase"] - 1) / n)
        st["hp"] = max(0, st["hp"] - dmg)
        st["log"].append(dict(phase=st["phase"], text=f"✔ {ph['title']}: {dmg} damage!"))
        st["phase"] += 1
        if ph["kind"] in ("mcq", "select"):
            item = dict(type="mcq", options=ph["q"]["options"], difficulty=ph["q"]["difficulty"]) if ph["kind"] == "mcq" else dict(type="experiment", difficulty=2)
        else:
            item = dict(type="experiment", difficulty=3)
        evidence(db, pid, concept, item, True, kind="boss", key=bid)
    else:
        st["mistakes"] += 1
        st["log"].append(dict(phase=st["phase"], text=f"✘ {ph['title']}: the boss shrugs it off."))
        if ph["kind"] == "mcq":
            evidence(db, pid, concept, dict(type="mcq", options=ph["q"]["options"], difficulty=ph["q"]["difficulty"]), False, kind="boss", key=bid)
    out["damage"] = dmg
    completed = st["phase"] >= len(b["phases"])
    if completed and not prog.get("completed_at"):
        xp = b["xp"] + max(0, 100 - 20 * st["mistakes"])
        db.add_xp(pid, xp, f"Boss defeated: {b['name']}")
        ach: list = []
        _grant(db, pid, f"boss_{bid}", ach)
        defeated = {k for k, v in db.progress(pid, "boss").items() if v.get("completed_at")} | {bid}
        if len(defeated) == len(bosses.BOSSES):
            _grant(db, pid, "all_bosses", ach)
        out.update(xp=xp, achievements=ach, lesson=b["lesson"])
    if completed:
        out["defeated"] = True
    db.set_progress(pid, "boss", bid, st, completed=completed)
    out["state"] = st
    return out


# ------------------------------------------------------------------ code dojo
def exercises_list(db: DB, pid: int) -> list[dict]:
    prog = db.progress(pid, "exercise")
    return [dict(id=e["id"], title=e["title"], concept=e["concept"], concept_name=CONCEPTS[e["concept"]]["name"], tier=e["tier"],
                 completed=bool(prog.get(e["id"], {}).get("completed_at")), attempts=prog.get(e["id"], {}).get("attempts", 0))
            for e in code_exercises.EXERCISES]


def exercise(db: DB, pid: int, ex_id: str) -> dict:
    e = code_exercises.BY_ID.get(ex_id)
    if not e:
        raise GameError("Unknown exercise", 404)
    st = db.progress(pid, "exercise").get(ex_id, {})
    reveal = bool(st.get("completed_at")) or st.get("attempts", 0) >= 5
    p = _player(db, pid)
    d = code_exercises.public(e, reveal_solution=reveal)
    if mastery.MODES[p["mode"]]["hints"] == "none":
        d["hints"] = []
    return dict(**d, state=dict(attempts=st.get("attempts", 0), completed=bool(st.get("completed_at")), code=st.get("code")))


def exercise_run(db: DB, pid: int, ex_id: str, code: str) -> dict:
    e = code_exercises.BY_ID.get(ex_id)
    if not e:
        raise GameError("Unknown exercise", 404)
    res = sandbox.run_tests(code, dict(setup=e.get("setup", ""), tests=e["tests"]))
    prog = db.progress(pid, "exercise").get(ex_id, {})
    st = {k: v for k, v in prog.items() if k != "completed_at"}
    st["attempts"] = st.get("attempts", 0) + 1
    st["code"] = code[:20000]
    passed = bool(res.get("ok")) and all(t["passed"] for t in res.get("tests", []))
    out = dict(**res, passed=passed)
    if passed and not prog.get("completed_at"):
        xp = 20 + 10 * e["tier"]
        db.add_xp(pid, xp, f"Code Dojo: {e['title']}")
        evidence(db, pid, e["concept"], dict(type="code", difficulty=3), True, kind="code", key=ex_id)
        ach: list = []
        _grant(db, pid, "code_first", ach)
        done = sum(1 for v in db.progress(pid, "exercise").values() if v.get("completed_at")) + 1
        if done >= 10:
            _grant(db, pid, "coder_10", ach)
        out.update(xp=xp, achievements=ach, solution=e["solution"])
    db.set_progress(pid, "exercise", ex_id, st, completed=passed)
    return out
