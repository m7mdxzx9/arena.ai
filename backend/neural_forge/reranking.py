"""Optional local reranking providers for a bounded retrieval shortlist."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Protocol, Sequence

import numpy as np

from .llm import ProviderError


class RerankerProvider(Protocol):
    def rerank(self, query: str, chunks: Sequence[dict[str, Any]]) -> list[float]: ...
    def health_check(self) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, Any]: ...


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _safe_model_name(name: str) -> str:
    value = str(name or "").strip()
    if not value or len(value) > 200 or ".." in value or value.startswith(("/", "\\")) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
        raise ProviderError("Choose a safe local reranker model identifier.", "invalid_reranker_model", 400)
    return value


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ProviderError(
            "Optional sentence-transformers support is not installed. The lexical overlap reranker is a heuristic, not a neural model.",
            "reranker_dependency_missing",
            503,
        ) from exc
    try:
        return CrossEncoder(model_name, device="cpu", local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise ProviderError("The cross-encoder is not available in the local model cache.", "reranker_model_not_cached", 503) from exc


class LocalCrossEncoderReranker:
    """Actual local cross-encoder scores query and passage pairs jointly."""

    def __init__(self, model: str = DEFAULT_CROSS_ENCODER):
        self.model = _safe_model_name(model)

    def rerank(self, query: str, chunks: Sequence[dict[str, Any]]) -> list[float]:
        if not chunks:
            return []
        model = _load_cross_encoder(self.model)
        values = model.predict([(query[:4_000], str(chunk.get("text", ""))[:10_000]) for chunk in chunks], show_progress_bar=False)
        scores = np.asarray(values, dtype=float).reshape(-1)
        if len(scores) != len(chunks) or not np.isfinite(scores).all():
            raise ProviderError("The local cross-encoder returned invalid scores.", "invalid_reranker_response", 502)
        return [float(score) for score in scores]

    def model_info(self) -> dict[str, Any]:
        return {"provider": "sentence-transformers-cross-encoder", "model": self.model, "device": "cpu", "local": True, "kind": "neural_cross_encoder"}

    def health_check(self) -> dict[str, Any]:
        try:
            self.rerank("health check", [{"text": "A short local reranker health check."}])
            return {**self.model_info(), "available": True}
        except ProviderError as exc:
            return {**self.model_info(), "available": False, "reason": exc.code}


def lexical_overlap_scores(query: str, chunks: Sequence[dict[str, Any]]) -> list[float]:
    """Cheap transparent fallback. It is deliberately named heuristic, never neural."""
    q = set(re.findall(r"[\w-]+", query.casefold()))
    if not q:
        return [0.0 for _ in chunks]
    scores = []
    for chunk in chunks:
        words = re.findall(r"[\w-]+", str(chunk.get("text", "")).casefold())
        counts = {word: words.count(word) for word in set(words)}
        coverage = sum(min(1, counts.get(term, 0)) for term in q) / len(q)
        bigrams = set(zip(re.findall(r"[\w-]+", query.casefold()), re.findall(r"[\w-]+", query.casefold())[1:]))
        text_bigrams = set(zip(words, words[1:]))
        phrase = len(bigrams & text_bigrams) / max(1, len(bigrams))
        scores.append(float(coverage + 0.2 * phrase))
    return scores


def apply_reranker(
    query: str,
    candidates: list[dict[str, Any]],
    reranker: str | None,
    depth: int = 20,
    *,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reranker in (None, "none", "off", ""):
        return candidates, {"enabled": False, "provider": None, "status": "disabled"}
    depth = max(1, min(int(depth), 100))
    short = candidates[:depth]
    if reranker == "lexical_overlap":
        scores = lexical_overlap_scores(query, short)
        label = "lexical_overlap_heuristic"
    elif reranker == "cross_encoder":
        provider = LocalCrossEncoderReranker(model or DEFAULT_CROSS_ENCODER)
        scores = provider.rerank(query, short)
        label = "local_neural_cross_encoder"
    else:
        raise ProviderError("Choose no reranker, lexical-overlap heuristic, or local cross-encoder.", "unknown_reranker", 400)
    for item, score in zip(short, scores):
        item["rerank_score"] = float(score)
        item["reranker"] = label
    short.sort(key=lambda item: (-float(item["rerank_score"]), int(item.get("rank", item.get("first_rank", 10**9)))))
    for position, item in enumerate(short, start=1):
        item["reranked_position"] = position
    tail = candidates[len(short) :]
    return short + tail, {"enabled": True, "provider": label, "status": "ready", "depth": depth, "model": model if reranker == "cross_encoder" else None}
