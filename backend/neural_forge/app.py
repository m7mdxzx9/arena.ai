"""FastAPI application: JSON API under /api and the built React frontend under /."""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import (
    agent,
    backup,
    capabilities,
    checkpoints,
    code_exercises,
    datasets,
    document_rag,
    evaluation,
    game,
    mastery,
    mistakes,
    ml,
    nn,
    playgrounds,
    portfolio,
    predictions,
    prompts,
    rag,
    real_agent,
    sandbox,
    torch_engine,
    tutor,
)
from .curriculum import AREAS, CONCEPTS, RANKS
from .db import DB
from .llm import ProviderError, get_provider
from .user_datasets import DatasetStore, DatasetUploadError

FRONTEND = Path(os.environ.get("NEURAL_FORGE_FRONTEND", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
LOGGER = logging.getLogger("neural_forge")


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
    language: str | None = None
    default_model: str | None = None


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


class TorchRun(BaseModel):
    config: dict = Field(default_factory=dict)
    name: str | None = None


class ModelChat(BaseModel):
    provider: str = "ollama"
    model: str
    messages: list[dict[str, str]]
    stream: bool = False


class ModelPullRequest(BaseModel):
    model: str = Field(min_length=1, max_length=160)
    provider: str = "ollama"


class TutorRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    mode: str = "simple"
    source: str = "offline"
    concept_id: str | None = None
    run_id: int | None = None
    model: str | None = None
    language: str = "en"


class MistakeReview(BaseModel):
    remembered: bool


class PersonalRagQuery(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    document_ids: list[str] | None = None
    method: str = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    generation: str = "extractive"
    model: str | None = None


class AgentConfigurationRequest(BaseModel):
    name: str = Field(default="Local Agent", max_length=100)
    model: str = Field(min_length=1, max_length=160)
    system_prompt: str = Field(default="You are a careful AI learning assistant.", max_length=4_000)
    tools: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=6, ge=1, le=12)
    timeout_seconds: float = Field(default=60, ge=5, le=180)


class AgentRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4_000)


class AgentArenaRequest(BaseModel):
    configuration_ids: list[str] = Field(min_length=2, max_length=4)
    tasks: list[dict[str, Any]] = Field(min_length=1, max_length=10)


class PromptRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PromptCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    system: str = Field(default="", max_length=10_000)
    user: str = Field(min_length=1, max_length=10_000)
    variables: dict[str, Any] = Field(default_factory=dict)


class PromptVersionCreate(BaseModel):
    system: str = Field(default="", max_length=10_000)
    user: str = Field(min_length=1, max_length=10_000)
    variables: dict[str, Any] = Field(default_factory=dict)
    change_note: str = Field(default="", max_length=500)


class PromptExecute(BaseModel):
    version: int = Field(ge=1)
    variables: dict[str, Any] = Field(default_factory=dict)
    model: str = Field(min_length=1, max_length=160)


class EvaluationDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    cases: list[dict[str, Any]] = Field(min_length=1, max_length=500)


class EvaluationRunRequest(BaseModel):
    outputs: dict[str, Any]
    name: str | None = Field(default=None, max_length=100)


class PortfolioCreateRequest(BaseModel):
    run_id: int = Field(ge=1)
    title: str = Field(default="", max_length=160)
    problem: str = Field(default="", max_length=10_000)
    dataset: str = Field(default="", max_length=2_000)
    method: str = Field(default="", max_length=10_000)
    metrics: Any = None
    interpretation: str = Field(default="", max_length=10_000)
    limitations: str = Field(default="", max_length=10_000)
    next_steps: str = Field(default="", max_length=10_000)


class PortfolioUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    problem: str | None = Field(default=None, max_length=10_000)
    dataset: str | None = Field(default=None, max_length=2_000)
    method: str | None = Field(default=None, max_length=10_000)
    metrics: Any = None
    interpretation: str | None = Field(default=None, max_length=10_000)
    limitations: str | None = Field(default=None, max_length=10_000)
    next_steps: str | None = Field(default=None, max_length=10_000)


class BackupRestoreRequest(BaseModel):
    backup: dict[str, Any]


class CheckpointUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


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
    dataset_store = DatasetStore(db)
    document_store = document_rag.DocumentStore(db)
    checkpoint_store = checkpoints.CheckpointStore(db)
    app.state.dataset_store = dataset_store
    app.state.document_store = document_store
    app.state.checkpoint_store = checkpoint_store

    @app.middleware("http")
    async def structured_request_log(request: Request, call_next):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                json.dumps({"event": "request_error", "request_id": request_id, "method": request.method, "path": request.url.path})
            )
            raise
        duration = round((time.perf_counter() - started) * 1000, 2)
        if request.url.path.startswith("/api/"):
            LOGGER.info(
                json.dumps(
                    {
                        "event": "api_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": duration,
                    },
                    separators=(",", ":"),
                )
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(game.GameError)
    async def _game_error(_: Request, exc: game.GameError):
        return SafeJSON({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(KeyError)
    async def _key_error(_: Request, exc: KeyError):
        return SafeJSON({"error": f"Unknown id: {exc}", "code": "not_found"}, status_code=404)

    @app.exception_handler(DatasetUploadError)
    async def _dataset_error(_: Request, exc: DatasetUploadError):
        return SafeJSON({"error": exc.message, "code": exc.code}, status_code=400)

    @app.exception_handler(document_rag.DocumentError)
    async def _document_error(_: Request, exc: document_rag.DocumentError):
        return SafeJSON({"error": exc.message, "code": exc.code}, status_code=400)

    @app.exception_handler(checkpoints.CheckpointError)
    async def _checkpoint_error(_: Request, exc: checkpoints.CheckpointError):
        status = 404 if exc.code == "checkpoint_missing" else 400
        return SafeJSON({"error": exc.message, "code": exc.code}, status_code=status)

    @app.exception_handler(ProviderError)
    async def _provider_error(_: Request, exc: ProviderError):
        return SafeJSON({"error": exc.message, "code": exc.code}, status_code=exc.status)

    @app.exception_handler(torch_engine.TorchUnavailableError)
    async def _torch_error(_: Request, exc: torch_engine.TorchUnavailableError):
        return SafeJSON({"error": str(exc), "code": "pytorch_unavailable"}, status_code=503)

    # ----------------------------------------------------------- static catalogue
    @app.get("/api/meta")
    def meta():
        return dict(modes={k: dict(name=v["name"], desc=v["desc"]) for k, v in mastery.MODES.items()}, areas=AREAS, ranks=[dict(title=r[0], xp=r[1], proficient=r[2], bosses=r[3]) for r in RANKS],
                    datasets=datasets.catalog(), models=ml.model_catalog(), predictions=predictions.catalog(),
                    agent=agent.catalog(), concepts={k: dict(name=v["name"], area=v["area"], branch=v["branch"]) for k, v in CONCEPTS.items()},
                    rag_corpus=rag.corpus_info())

    # ----------------------------------------------------------- runtime capabilities & local models

    @app.get("/api/system/capabilities")
    def system_capabilities(refresh: bool = False):
        return capabilities.system_capabilities(force=refresh)

    @app.get("/api/models")
    def local_models(provider: str = "ollama"):
        selected = get_provider(provider)
        health = selected.health_check()
        if not health["reachable"]:
            return {"provider": provider, "reachable": False, "models": [], "error": health.get("error")}
        return {"provider": provider, "reachable": True, "models": selected.list_models()}

    @app.get("/api/models/{model_name:path}")
    def local_model_info(model_name: str, provider: str = "ollama"):
        return get_provider(provider).model_info(model_name)

    @app.post("/api/models/pull")
    def local_model_pull(body: ModelPullRequest):
        return get_provider(body.provider).pull_model(body.model)

    @app.delete("/api/models/{model_name:path}")
    def local_model_delete(model_name: str, provider: str = "ollama"):
        return get_provider(provider).delete_model(model_name)

    @app.post("/api/models/chat")
    def local_model_chat(body: ModelChat):
        selected = get_provider(body.provider)
        if not body.stream:
            return selected.chat(body.model, body.messages)

        def events():
            for event in selected.stream_chat(body.model, body.messages):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

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
        return game.set_settings(
            db,
            pid,
            body.mode,
            body.free_play,
            body.show_code,
            body.language,
            body.default_model,
        )

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

    # Personal Dataset Workspace. Client filenames are display metadata only; all
    # storage paths are generated server-side by DatasetStore.
    @app.get("/api/p/{pid}/datasets")
    def personal_datasets(pid: int):
        game._player(db, pid)
        return dataset_store.list(pid)

    @app.post("/api/p/{pid}/datasets")
    async def upload_dataset(pid: int, file: UploadFile = File(...), name: str | None = Form(default=None)):
        game._player(db, pid)
        # SpooledUploadFile keeps bounded uploads out of application memory when
        # possible; parse_upload independently enforces the hard byte limit.
        return dataset_store.create_from_upload(pid, file.file, file.filename or "dataset", name, file.size)

    @app.get("/api/p/{pid}/datasets/{dataset_id}")
    def personal_dataset(pid: int, dataset_id: str, target: str | None = None):
        return dataset_store.profile(pid, dataset_id, target)

    @app.delete("/api/p/{pid}/datasets/{dataset_id}")
    def delete_personal_dataset(pid: int, dataset_id: str):
        if not dataset_store.delete(pid, dataset_id):
            raise game.GameError("Uploaded dataset not found", 404)
        return {"ok": True}

    @app.get("/api/p/{pid}/datasets/{dataset_id}/scatter")
    def personal_dataset_scatter(pid: int, dataset_id: str, x: str, y: str, color: str | None = None):
        from . import datalab
        frame = dataset_store.load(pid, dataset_id)
        for column in (x, y, color):
            if column and column not in frame.columns:
                raise game.GameError(f"Unknown column {column}")
        return datalab.scatter(frame, x, y, color)

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

    # ----------------------------------------------------------- professional PyTorch / CNN labs

    def _discard_failed_training(pid: int, checkpoint_id: str, path: Path, run_id: int | None) -> None:
        try:
            checkpoint_store.discard(pid, checkpoint_id, path)
        except Exception:
            LOGGER.exception("Failed to clean checkpoint after a training error")
        if run_id is not None:
            db.x("DELETE FROM runs WHERE id=? AND player_id=?", (run_id, pid))

    @app.post("/api/p/{pid}/pytorch/train")
    def pytorch_train(pid: int, body: TorchRun):
        game._player(db, pid)
        checkpoint_id, path = checkpoint_store.allocate(pid)
        run_id: int | None = None
        dataset_id = str(body.config.get("dataset", ""))
        frame = dataset_store.load(pid, dataset_id.removeprefix("user:")) if dataset_id.startswith("user:") else None
        try:
            result = torch_engine.train_tabular(body.config, path, frame)
            summary = {"engine": "pytorch", "device": result["device"], "final": result["final"], "duration_seconds": result["duration_seconds"]}
            run_id = db.save_run(pid, "pytorch", result["config"], result, summary, name=body.name, context="open_lab")
            checkpoint = checkpoint_store.register(pid, checkpoint_id, path, run_id, "pytorch", body.name, {"device": result["device"], "final": result["final"]})
        except ValueError as exc:
            _discard_failed_training(pid, checkpoint_id, path, run_id)
            raise game.GameError(str(exc)) from exc
        except Exception:
            _discard_failed_training(pid, checkpoint_id, path, run_id)
            raise
        result["checkpoint"] = checkpoint
        return {"ok": True, "run_id": run_id, "checkpoint_id": checkpoint_id, "checkpoint": checkpoint, "result": result, **game._after_run(db, pid)}

    @app.post("/api/p/{pid}/cnn/train")
    def cnn_train(pid: int, body: TorchRun):
        game._player(db, pid)
        checkpoint_id, path = checkpoint_store.allocate(pid)
        run_id: int | None = None
        try:
            result = torch_engine.train_cnn(body.config, path)
            summary = {"engine": "pytorch", "device": result["device"], "final": result["final"], "duration_seconds": result["duration_seconds"]}
            run_id = db.save_run(pid, "cnn", result["config"], result, summary, name=body.name, context="open_lab")
            checkpoint = checkpoint_store.register(pid, checkpoint_id, path, run_id, "cnn", body.name, {"device": result["device"], "final": result["final"]})
        except ValueError as exc:
            _discard_failed_training(pid, checkpoint_id, path, run_id)
            raise game.GameError(str(exc)) from exc
        except Exception:
            _discard_failed_training(pid, checkpoint_id, path, run_id)
            raise
        result["checkpoint"] = checkpoint
        return {"ok": True, "run_id": run_id, "checkpoint_id": checkpoint_id, "checkpoint": checkpoint, "result": result, **game._after_run(db, pid)}

    @app.get("/api/p/{pid}/checkpoints")
    def checkpoint_list(pid: int):
        game._player(db, pid)
        return checkpoint_store.list(pid)

    @app.get("/api/p/{pid}/checkpoints/{checkpoint_id}")
    def checkpoint_detail(pid: int, checkpoint_id: str):
        return checkpoint_store.get(pid, checkpoint_id)

    @app.patch("/api/p/{pid}/checkpoints/{checkpoint_id}")
    def checkpoint_update(pid: int, checkpoint_id: str, body: CheckpointUpdateRequest):
        return checkpoint_store.rename(pid, checkpoint_id, body.name)

    @app.get("/api/p/{pid}/checkpoints/{checkpoint_id}/download")
    def checkpoint_download(pid: int, checkpoint_id: str):
        path = checkpoint_store.download_path(pid, checkpoint_id)
        return FileResponse(path, media_type="application/octet-stream", filename=f"neural-forge-{checkpoint_id}.pt")

    @app.delete("/api/p/{pid}/checkpoints/{checkpoint_id}")
    def checkpoint_delete(pid: int, checkpoint_id: str):
        return checkpoint_store.delete(pid, checkpoint_id)

    # ----------------------------------------------------------- tutor & mistake journal

    @app.post("/api/p/{pid}/tutor")
    def ai_tutor(pid: int, body: TutorRequest):
        game._player(db, pid)
        try:
            if body.source == "offline":
                return tutor.curated_answer(db, pid, body.question, body.mode, concept_id=body.concept_id, run_id=body.run_id, language=body.language)
            if body.source == "ollama":
                if not body.model:
                    raise game.GameError("Select a local model first.")
                return tutor.local_llm_answer(db, pid, body.question, body.mode, body.model, concept_id=body.concept_id, run_id=body.run_id, language=body.language)
            raise game.GameError("Unknown tutor source.")
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/mistakes")
    def mistake_records(pid: int, status: str = "all", topic: str | None = None):
        game._player(db, pid)
        if status not in {"all", "unresolved", "mastered", "due"}:
            raise game.GameError("Unknown mistake filter.")
        return mistakes.list_records(db, pid, status, topic)

    @app.post("/api/p/{pid}/mistakes/{mistake_id}/review")
    def review_mistake(pid: int, mistake_id: int, body: MistakeReview):
        try:
            return mistakes.review(db, pid, mistake_id, body.remembered)
        except KeyError as exc:
            raise game.GameError("Mistake record not found", 404) from exc

    # ----------------------------------------------------------- personal-document RAG

    @app.get("/api/p/{pid}/documents")
    def documents(pid: int):
        game._player(db, pid)
        return document_store.list(pid)

    @app.post("/api/p/{pid}/documents")
    async def upload_document(
        pid: int,
        file: UploadFile = File(...),
        name: str | None = Form(default=None),
        chunk_size: int = Form(default=180),
        overlap: int = Form(default=30),
    ):
        game._player(db, pid)
        return document_store.add(pid, file.file, file.filename or "document", name, chunk_size, overlap)

    @app.delete("/api/p/{pid}/documents/{document_id}")
    def delete_document(pid: int, document_id: str):
        if not document_store.delete(pid, document_id):
            raise game.GameError("Document not found", 404)
        return {"ok": True}

    @app.post("/api/p/{pid}/personal-rag/query")
    def personal_rag_query(pid: int, body: PersonalRagQuery):
        game._player(db, pid)
        return document_rag.query(
            db,
            pid,
            body.question,
            document_ids=body.document_ids,
            method=body.method,
            top_k=body.top_k,
            generation=body.generation,
            model=body.model,
        )

    # ----------------------------------------------------------- real local-model agent lab

    @app.get("/api/agent-tools")
    def agent_tools():
        return real_agent.catalog()

    @app.get("/api/p/{pid}/agent-configurations")
    def agent_configurations(pid: int):
        game._player(db, pid)
        return real_agent.list_configurations(db, pid)

    @app.post("/api/p/{pid}/agent-configurations")
    def create_agent_configuration(pid: int, body: AgentConfigurationRequest):
        game._player(db, pid)
        try:
            return real_agent.create_configuration(db, pid, body.model_dump())
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.patch("/api/p/{pid}/agent-configurations/{configuration_id}")
    def update_agent_configuration(pid: int, configuration_id: str, body: AgentConfigurationRequest):
        try:
            return real_agent.update_configuration(db, pid, configuration_id, body.model_dump())
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.delete("/api/p/{pid}/agent-configurations/{configuration_id}")
    def delete_agent_configuration(pid: int, configuration_id: str):
        return real_agent.delete_configuration(db, pid, configuration_id)

    @app.post("/api/p/{pid}/agent-configurations/{configuration_id}/clear-memory")
    def clear_agent_memory(pid: int, configuration_id: str):
        try:
            return real_agent.clear_memory(db, pid, configuration_id)
        except KeyError as exc:
            raise game.GameError("Agent configuration not found", 404) from exc

    @app.post("/api/p/{pid}/agent-configurations/{configuration_id}/run")
    def run_real_agent(pid: int, configuration_id: str, body: AgentRunRequest):
        try:
            return real_agent.run_agent(db, pid, configuration_id, body.request)
        except KeyError as exc:
            raise game.GameError("Agent configuration not found", 404) from exc
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/agent-runs")
    def real_agent_runs(pid: int, limit: int = 50):
        game._player(db, pid)
        return real_agent.list_runs(db, pid, limit)

    @app.post("/api/p/{pid}/agent-arena")
    def real_agent_arena(pid: int, body: AgentArenaRequest):
        game._player(db, pid)
        try:
            return real_agent.run_arena(db, pid, body.configuration_ids, body.tasks)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    # ----------------------------------------------------------- prompt & central evaluation labs

    @app.get("/api/p/{pid}/prompts")
    def prompt_list(pid: int):
        game._player(db, pid)
        return prompts.list_prompts(db, pid)

    @app.post("/api/p/{pid}/prompts")
    def prompt_create(pid: int, body: PromptCreate):
        game._player(db, pid)
        try:
            return prompts.create(db, pid, body.name, body.system, body.user, body.variables)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/prompts/{prompt_id}")
    def prompt_detail(pid: int, prompt_id: str):
        return prompts.get(db, pid, prompt_id)

    @app.patch("/api/p/{pid}/prompts/{prompt_id}")
    def prompt_rename(pid: int, prompt_id: str, body: PromptRenameRequest):
        try:
            return prompts.rename(db, pid, prompt_id, body.name)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.delete("/api/p/{pid}/prompts/{prompt_id}")
    def prompt_delete(pid: int, prompt_id: str):
        return prompts.delete(db, pid, prompt_id)

    @app.post("/api/p/{pid}/prompts/{prompt_id}/versions")
    def prompt_add_version(pid: int, prompt_id: str, body: PromptVersionCreate):
        try:
            return prompts.add_version(db, pid, prompt_id, body.system, body.user, body.variables, body.change_note)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.post("/api/p/{pid}/prompts/{prompt_id}/execute")
    def prompt_execute(pid: int, prompt_id: str, body: PromptExecute):
        try:
            return prompts.execute(db, pid, prompt_id, body.version, body.variables, body.model)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/evaluation-datasets")
    def evaluation_dataset_list(pid: int):
        game._player(db, pid)
        return evaluation.list_datasets(db, pid)

    @app.post("/api/p/{pid}/evaluation-datasets")
    def evaluation_dataset_create(pid: int, body: EvaluationDatasetCreate):
        game._player(db, pid)
        try:
            return evaluation.create_dataset(db, pid, body.name, body.cases)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/evaluation-datasets/{dataset_id}")
    def evaluation_dataset_detail(pid: int, dataset_id: str):
        return evaluation.get_dataset(db, pid, dataset_id)

    @app.put("/api/p/{pid}/evaluation-datasets/{dataset_id}")
    def evaluation_dataset_update(pid: int, dataset_id: str, body: EvaluationDatasetCreate):
        try:
            return evaluation.update_dataset(db, pid, dataset_id, body.name, body.cases)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.delete("/api/p/{pid}/evaluation-datasets/{dataset_id}")
    def evaluation_dataset_delete(pid: int, dataset_id: str):
        return evaluation.delete_dataset(db, pid, dataset_id)

    @app.post("/api/p/{pid}/evaluation-datasets/{dataset_id}/run")
    def evaluation_run(pid: int, dataset_id: str, body: EvaluationRunRequest):
        try:
            return evaluation.run_evaluation(db, pid, dataset_id, body.outputs, body.name)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    # ----------------------------------------------------------- portfolio and portable backups

    @app.get("/api/p/{pid}/portfolio")
    def portfolio_list(pid: int):
        game._player(db, pid)
        return portfolio.list_projects(db, pid)

    @app.post("/api/p/{pid}/portfolio")
    def portfolio_create(pid: int, body: PortfolioCreateRequest):
        game._player(db, pid)
        try:
            values = body.model_dump()
            run_id = values.pop("run_id")
            return portfolio.create_from_run(db, pid, run_id, values)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.patch("/api/p/{pid}/portfolio/{project_id}")
    def portfolio_update(pid: int, project_id: str, body: PortfolioUpdateRequest):
        try:
            return portfolio.update_project(db, pid, project_id, body.model_dump(exclude_unset=True))
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.delete("/api/p/{pid}/portfolio/{project_id}")
    def portfolio_delete(pid: int, project_id: str):
        return portfolio.delete_project(db, pid, project_id)

    @app.get("/api/p/{pid}/portfolio/{project_id}/export")
    def portfolio_export(pid: int, project_id: str, format: str = "markdown"):
        try:
            content, media_type = portfolio.export_project(portfolio.get_project(db, pid, project_id), format)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc
        extension = {"markdown": "md", "html": "html", "json": "json"}.get(format, "txt")
        return PlainTextResponse(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="neural-forge-{project_id}.{extension}"'})

    @app.get("/api/p/{pid}/backup")
    def backup_export(pid: int):
        try:
            payload = backup.export_player(db, pid)
        except ValueError as exc:
            raise game.GameError(str(exc), 413) from exc
        return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="neural-forge-player-{pid}.json"'})

    @app.post("/api/p/{pid}/restore")
    def backup_restore(pid: int, body: BackupRestoreRequest):
        try:
            return backup.restore_player(db, pid, body.backup)
        except ValueError as exc:
            raise game.GameError(str(exc)) from exc

    @app.get("/api/p/{pid}/backup/full")
    def full_backup_export(pid: int):
        game._player(db, pid)
        temporary = tempfile.NamedTemporaryFile(prefix=f"neural-forge-{pid}-", suffix=".nfbackup", delete=False)
        path = Path(temporary.name)
        temporary.close()
        try:
            backup.write_full_archive(db, pid, path, dataset_store, document_store, checkpoint_store)
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise game.GameError(str(exc), 413) from exc
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"neural-forge-player-{pid}.nfbackup",
            background=BackgroundTask(path.unlink, missing_ok=True),
        )

    @app.post("/api/p/{pid}/restore/full")
    async def full_backup_restore(pid: int, file: UploadFile = File(...)):
        game._player(db, pid)
        temporary = tempfile.NamedTemporaryFile(prefix="neural-forge-restore-", suffix=".nfbackup", delete=False)
        path = Path(temporary.name)
        size = 0
        try:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > backup.MAX_ARCHIVE_BYTES:
                    raise game.GameError("Full backup archive exceeds the 2 GiB upload limit.", 413)
                temporary.write(chunk)
            temporary.close()
            if not size:
                raise game.GameError("Full backup archive is empty.")
            return backup.restore_full_archive(db, pid, path, dataset_store, document_store, checkpoint_store)
        except game.GameError:
            raise
        except (ValueError, zipfile.BadZipFile, OSError) as exc:
            raise game.GameError(str(exc) or "Full backup archive could not be restored.") from exc
        finally:
            if not temporary.closed:
                temporary.close()
            path.unlink(missing_ok=True)

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
