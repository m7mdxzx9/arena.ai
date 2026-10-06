"""FastAPI application: JSON API under /api and the built React frontend under /."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agent, code_exercises, mastery, datasets, game, ml, nn, playgrounds, predictions, rag, sandbox
from .curriculum import AREAS, CONCEPTS, RANKS
from .db import DB

FRONTEND = Path(os.environ.get("NEURAL_FORGE_FRONTEND", Path(__file__).resolve().parents[2] / "frontend" / "dist"))


def _clean(o: Any):
    """Make results strictly JSON-safe (NaN/inf → null, numpy → python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


class SafeJSON(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(_clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


class NewPlayer(BaseModel):
    name: str = "Apprentice"
    mode: int = 1


class Settings(BaseModel):
    mode: int | None = None
    free_play: bool | None = None
    show_code: bool | None = None


class Answer(BaseModel):
    key: str
    response: Any = None
    hint_used: bool = False
    purpose: str = "practice"


class Text(BaseModel):
    text: str


class Predict(BaseModel):
    choice: int


class RunReq(BaseModel):
    config: dict = Field(default_factory=dict)
    context: str | None = None
    name: str | None = None


class RunUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None


class Compare(BaseModel):
    run_ids: list[int]


class Action(BaseModel):
    action: str
    payload: dict = Field(default_factory=dict)


class Submit(BaseModel):
    run_id: int


class Code(BaseModel):
    code: str


BOUNDARY_MODELS = {"logistic_regression", "decision_tree", "random_forest", "knn", "naive_bayes", "gradient_boosting"}


def _boundary(params: dict) -> dict:
    ds = params.get("dataset", "moons")
    model = params.get("model", "logistic_regression")
    if ds not in ("moons", "circles", "linear2d", "xor", "spiral", "blobs") or model not in BOUNDARY_MODELS:
        raise game.GameError("Unsupported dataset/model for this widget")
    hp = {}
    if "max_depth" in params:
        hp["max_depth"] = int(params["max_depth"])
    if "n_neighbors" in params:
        hp["n_neighbors"] = int(params["n_neighbors"])
    tgt = datasets.REGISTRY[ds]["target"]
    feats = [c for c in datasets.load(ds).columns if c != tgt]
    r = ml.run_experiment(ml.Experiment(dataset=ds, target=tgt, features=feats, model=model, params=hp))
    return dict(boundary=r["decision_boundary"], train=r["metrics"]["train"], test=r["metrics"]["test"], model=ml.MODELS[model]["label"], params=hp)


def _histogram(params: dict) -> dict:
    from . import datalab
    ds = params.get("dataset", "house_prices")
    df = datasets.load(ds)
    col = params.get("column") or next(c for c in df.select_dtypes("number").columns)
    if col not in df.columns:
        raise game.GameError("Unknown column")
    s = df[col]
    return dict(dataset=ds, column=col, columns=df.select_dtypes("number").columns.tolist(), hist=datalab.histogram(s, int(params.get("bins", 20))),
                mean=float(s.mean()), median=float(s.median()), std=float(s.std()))


def _imbalance() -> dict:
    df = datasets.load("fraud")
    vc = df["is_fraud"].value_counts()
    return dict(counts={str(k): int(v) for k, v in vc.items()}, majority_share=float(vc.max() / vc.sum()), n=int(vc.sum()))


def _missing() -> dict:
    df = datasets.load("student_success").head(80)
    return dict(columns=df.columns.tolist(), missing=df.isna().astype(int).values.T.tolist(), totals=datasets.load("student_success").isna().sum().astype(int).to_dict())


def _peek(ds: str) -> dict:
    if ds not in datasets.REGISTRY:
        raise game.GameError("Unknown dataset")
    df = datasets.load(ds)
    return dict(dataset=ds, target=datasets.REGISTRY[ds]["target"], columns=df.columns.tolist()[:10], rows=df.head(8).iloc[:, :10].astype(object).where(df.head(8).iloc[:, :10].notna(), None).values.tolist(),
                n_rows=len(df), n_cols=df.shape[1])


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="NEURAL FORGE", default_response_class=SafeJSON)
    db = DB(db_path)
    app.state.db = db

    @app.exception_handler(game.GameError)
    async def _game_error(_: Request, exc: game.GameError):
        return SafeJSON({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(KeyError)
    async def _key_error(_: Request, exc: KeyError):
        return SafeJSON({"error": f"Unknown id: {exc}"}, status_code=404)

    # ----------------------------------------------------------- static catalogue
    @app.get("/api/meta")
    def meta():
        return dict(modes={k: dict(name=v["name"], desc=v["desc"]) for k, v in mastery.MODES.items()}, areas=AREAS, ranks=[dict(title=r[0], xp=r[1], proficient=r[2], bosses=r[3]) for r in RANKS],
                    datasets=datasets.catalog(), models=ml.model_catalog(), predictions=predictions.catalog(),
                    agent=agent.catalog(), concepts={k: dict(name=v["name"], area=v["area"], branch=v["branch"]) for k, v in CONCEPTS.items()},
                    rag_corpus=rag.corpus_info())

    # ----------------------------------------------------------- players


    @app.get("/api/players")
    def players():
        return db.players()

    @app.post("/api/players")
    def new_player(body: NewPlayer):
        return game.create_player(db, body.name, body.mode)

    @app.get("/api/p/{pid}")
    def overview(pid: int):
        return game.overview(db, pid)

    @app.post("/api/p/{pid}/settings")
    def settings(pid: int, body: Settings):
        return game.set_settings(db, pid, body.mode, body.free_play, body.show_code)

    @app.get("/api/p/{pid}/tree")
    def tree(pid: int):
        return game.tree(db, pid)

    @app.get("/api/p/{pid}/review")
    def review(pid: int):
        return game.review_queue(db, pid)

    # ----------------------------------------------------------- lessons


    @app.get("/api/p/{pid}/lesson/{cid}")
    def lesson(pid: int, cid: str):
        return game.lesson(db, pid, cid)

    @app.get("/api/p/{pid}/lesson/{cid}/question")
    def question(pid: int, cid: str, purpose: str = "practice"):
        return game.next_question(db, pid, cid, purpose)

    @app.get("/api/p/{pid}/lesson/{cid}/hint")
    def hint(pid: int, cid: str, key: str):
        return game.hint(db, pid, cid, key)

    @app.post("/api/p/{pid}/lesson/{cid}/answer")
    def answer(pid: int, cid: str, body: Answer):
        return game.answer(db, pid, cid, body.key, body.response, body.hint_used, body.purpose)

    @app.post("/api/p/{pid}/lesson/{cid}/reflect")
    def reflect(pid: int, cid: str, body: Text):
        return game.reflect(db, pid, cid, body.text)

    @app.get("/api/p/{pid}/reflections")
    def reflections(pid: int):
        return db.reflections(pid)

    # ----------------------------------------------------------- predictions

    @app.post("/api/p/{pid}/predict/{exp_id}")
    def predict(pid: int, exp_id: str, body: Predict):
        return game.prediction(db, pid, exp_id, body.choice)

    # ----------------------------------------------------------- labs & experiment history

    @app.get("/api/p/{pid}/data/{ds_id}")
    def data_profile(pid: int, ds_id: str):
        return game.profile(db, pid, ds_id)

    @app.get("/api/data/{ds_id}/scatter")
    def data_scatter(ds_id: str, x: str, y: str, color: str | None = None):
        from . import datalab
        df = datasets.load(ds_id)
        for c in (x, y, color):
            if c and c not in df.columns:
                raise game.GameError(f"Unknown column {c}")
        return datalab.scatter(df, x, y, color)

    @app.post("/api/p/{pid}/run/{kind}")
    def run(pid: int, kind: str, body: RunReq):
        if kind not in game.RUNNERS:
            raise game.GameError("Unknown lab")
        return game.RUNNERS[kind](db, pid, body.config, body.context, body.name)

    @app.get("/api/p/{pid}/runs")
    def runs(pid: int, kind: str | None = None):
        return db.runs(pid, kind)

    @app.get("/api/p/{pid}/runs/{rid}")
    def run_detail(pid: int, rid: int):
        r = db.run(pid, rid)
        if not r:
            raise game.GameError("Run not found", 404)
        return r


    @app.post("/api/p/{pid}/runs/{rid}")
    def run_update(pid: int, rid: int, body: RunUpdate):
        fields = {k: v[:2000] for k, v in body.model_dump().items() if v is not None}
        if fields:
            db.update_run(pid, rid, **fields)
        out: list = []
        if fields.get("notes"):
            game._grant(db, pid, "reproducible", out)
        return dict(ok=True, achievements=out)


    @app.post("/api/p/{pid}/compare")
    def compare(pid: int, body: Compare):
        return game.compare(db, pid, body.run_ids)

    @app.get("/api/nn/source", response_class=PlainTextResponse)
    def nn_source():
        return nn.source()

    @app.get("/api/rag/map")
    def rag_map(chunk_size: int = 60, overlap: int = 10, embedding: str = "lsa", q: str | None = None):
        return rag.embedding_map(q, dict(chunk_size=chunk_size, overlap=overlap, embedding=embedding))

    # ----------------------------------------------------------- missions, challenges, bosses

    @app.get("/api/p/{pid}/missions")
    def missions_list(pid: int):
        return game.missions_list(db, pid)

    @app.get("/api/p/{pid}/missions/{mid}")
    def mission(pid: int, mid: str):
        return game.mission(db, pid, mid)

    @app.post("/api/p/{pid}/missions/{mid}")
    def mission_action(pid: int, mid: str, body: Action):
        return game.mission_action(db, pid, mid, body.action, body.payload)

    @app.get("/api/p/{pid}/challenges")
    def challenges(pid: int):
        return game.challenges_list(db, pid)


    @app.post("/api/p/{pid}/challenges/{ch_id}")
    def challenge_submit(pid: int, ch_id: str, body: Submit):
        return game.challenge_submit(db, pid, ch_id, body.run_id)

    @app.get("/api/p/{pid}/bosses")
    def bosses_list(pid: int):
        return game.bosses_list(db, pid)

    @app.get("/api/p/{pid}/bosses/{bid}")
    def boss(pid: int, bid: str):
        return game.boss(db, pid, bid)

    @app.post("/api/p/{pid}/bosses/{bid}")
    def boss_action(pid: int, bid: str, body: dict):
        return game.boss_action(db, pid, bid, body)

    # ----------------------------------------------------------- code dojo

    @app.get("/api/p/{pid}/exercises")
    def exercises(pid: int):
        return game.exercises_list(db, pid)

    @app.get("/api/p/{pid}/exercises/{ex_id}")
    def exercise(pid: int, ex_id: str):
        return game.exercise(db, pid, ex_id)

    @app.post("/api/p/{pid}/exercises/{ex_id}")
    def exercise_run(pid: int, ex_id: str, body: Code):
        return game.exercise_run(db, pid, ex_id, body.code)

    @app.post("/api/code/run")
    def code_run(body: Code):
        return sandbox.run_free(body.code)

    @app.post("/api/code/trace")
    def code_trace(body: Code):
        return sandbox.trace(body.code)

    # ----------------------------------------------------------- interactive widgets (real computations)
    @app.get("/api/widget/{name}")
    def widget(name: str, request: Request):
        params = dict(request.query_params)
        fns = {
            "tokenizer": lambda: playgrounds.tokenizer_demo(params.get("text", "unbelievable tokenization"), int(params.get("merges", 60))),
            "attention": lambda: playgrounds.attention_demo(params.get("sentence", "the cat sat on the mat because it was tired")),
            "kmeans": lambda: playgrounds.kmeans_steps(params.get("dataset", "blobs"), int(params.get("k", 3)), int(params.get("seed", 0))),
            "linreg": lambda: playgrounds.linreg_points(int(params.get("seed", 3))),
            "split": lambda: playgrounds.split_demo(float(params.get("test_size", 0.2)), int(params.get("seed", 42)), params.get("stratify", "true") == "true"),
            "cv": lambda: playgrounds.cv_demo(int(params.get("folds", 5))),
            "overfit": lambda: playgrounds.overfit_curve(params.get("dataset", "student_success")),
            "threshold": lambda: playgrounds.threshold_data(params.get("dataset", "fraud")),
            "conv": lambda: playgrounds.conv_demo(int(params.get("index", 0)), params.get("kernel", "vertical_edge")),
            "lm": lambda: playgrounds.lm_demo(params.get("word", "the"), float(params.get("temperature", 1.0)), int(params.get("seed", 0))),
            "distribution": lambda: playgrounds.distribution_samples(params.get("kind", "normal")),
            "onehot": playgrounds.onehot_demo,
            "scaling": playgrounds.scaling_demo,
            "chunks": lambda: dict(chunks=rag.chunk_docs(int(params.get("chunk_size", 60)), int(params.get("overlap", 10)))[:12]),
            "boundary": lambda: _boundary(params),
            "histogram": lambda: _histogram(params),
            "imbalance": lambda: _imbalance(),
            "missing": lambda: _missing(),
            "peek": lambda: _peek(params.get("dataset", "student_success")),
        }
        if name not in fns:
            raise game.GameError("Unknown widget", 404)
        try:
            return fns[name]()
        except (ValueError, KeyError) as e:
            raise game.GameError(f"Bad widget parameters: {e}")

    @app.get("/api/health")
    def health():
        return dict(ok=True)

    # ----------------------------------------------------------- frontend
    if FRONTEND.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = FRONTEND / path
            if path and f.is_file() and FRONTEND in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(FRONTEND / "index.html")

    return app


app = create_app()
