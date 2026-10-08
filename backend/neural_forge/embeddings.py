"""Local embedding providers for the personal RAG workspace.

Statistical TF-IDF/LSA lives in ``document_rag`` and is deliberately not called a
neural embedding. The providers here execute actual embedding models locally:
Sentence Transformers (CPU, cached model only) or an Ollama embedding model on the
configured local Ollama service. No provider sends data to a hosted API.
"""
from __future__ import annotations

import importlib.util
import re
from functools import lru_cache
from typing import Any, Protocol, Sequence

import numpy as np

from .llm import LLMProvider, ProviderError, get_provider

DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_EMBEDDING = "nomic-embed-text"
BATCH_SIZE = 32


class EmbeddingProvider(Protocol):
    """Contract shared by local embedding implementations."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def health_check(self) -> dict[str, Any]: ...
    def model_info(self) -> dict[str, Any]: ...
    def dimension(self) -> int | None: ...


def _safe_model_name(name: str) -> str:
    value = str(name or "").strip()
    if not value or len(value) > 200 or ".." in value or value.startswith(("/", "\\")):
        raise ProviderError("Choose a safe, local embedding model identifier.", "invalid_embedding_model", 400)
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
        raise ProviderError("Embedding model identifiers may contain only letters, numbers and . _ : / -.", "invalid_embedding_model", 400)
    return value


def _normalise(vectors: Any, expected: int | None = None) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0 or (expected is not None and len(array) != expected):
        raise ProviderError("The local embedding model returned an invalid vector shape.", "invalid_embedding_response", 502)
    if not np.isfinite(array).all():
        raise ProviderError("The local embedding model returned a non-finite vector.", "invalid_embedding_response", 502)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


class OllamaEmbeddingProvider:
    """Actual vectors from Ollama's local ``/api/embed`` endpoint, in bounded batches."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_EMBEDDING, client: LLMProvider | None = None):
        self.model = _safe_model_name(model)
        self.client = client or get_provider("ollama")
        self._dimension: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        output: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [str(text)[:20_000] for text in texts[start : start + BATCH_SIZE]]
            try:
                output.extend(self.client.embeddings(self.model, batch))
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError("The local Ollama embedding request failed.", "embedding_unavailable", 503) from exc
        vectors = _normalise(output, len(texts))
        self._dimension = int(vectors.shape[1])
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def dimension(self) -> int | None:
        return self._dimension

    def model_info(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dimension": self._dimension, "device": "local Ollama", "local": True}

    def health_check(self) -> dict[str, Any]:
        try:
            health = self.client.health_check()
            if not health.get("reachable"):
                return {**self.model_info(), "available": False, "reason": health.get("code", "ollama_unavailable")}
            models = self.client.list_models()
            installed = any(item.get("name") == self.model for item in models)
            return {**self.model_info(), "available": installed, "reason": None if installed else "embedding_model_not_installed"}
        except Exception as exc:
            return {**self.model_info(), "available": False, "reason": getattr(exc, "code", "ollama_unavailable")}


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    # Model downloads are intentionally disabled. Students must install/cache the
    # open model explicitly; uploaded academic text never leaves this machine.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ProviderError(
            "Optional sentence-transformers support is not installed. Install requirements-rag.txt or use a local Ollama embedding model.",
            "embedding_dependency_missing",
            503,
        ) from exc
    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise ProviderError(
            "The selected Sentence Transformers model is not available in the local cache. Downloads are disabled by default.",
            "embedding_model_not_cached",
            503,
        ) from exc


class SentenceTransformerEmbeddingProvider:
    """Small, CPU-capable Sentence Transformers model loaded lazily from local cache."""

    name = "sentence-transformers"

    def __init__(self, model: str = DEFAULT_SENTENCE_TRANSFORMER, batch_size: int = BATCH_SIZE):
        self.model = _safe_model_name(model)
        self.batch_size = max(1, min(int(batch_size), 128))
        self._dimension: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension or 0), dtype=np.float32)
        model = _load_sentence_transformer(self.model)
        vectors = model.encode(
            [str(text)[:20_000] for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        output = _normalise(vectors, len(texts))
        self._dimension = int(output.shape[1])
        return output

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def dimension(self) -> int | None:
        return self._dimension

    def model_info(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dimension": self._dimension, "device": "cpu", "local": True}

    def health_check(self) -> dict[str, Any]:
        try:
            self.embed_query("local embedding health check")
            return {**self.model_info(), "available": True, "reason": None}
        except ProviderError as exc:
            return {**self.model_info(), "available": False, "reason": exc.code}


def create_embedding_provider(provider: str, model: str | None = None, *, llm_provider: LLMProvider | None = None) -> EmbeddingProvider:
    """Create a named local provider. ``statistical_lsa`` is not an embedding provider."""
    if provider == "ollama":
        return OllamaEmbeddingProvider(model or DEFAULT_OLLAMA_EMBEDDING, llm_provider)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(model or DEFAULT_SENTENCE_TRANSFORMER)
    raise ProviderError(f"Unknown local embedding provider '{provider}'.", "unknown_embedding_provider", 400)


def provider_capabilities() -> list[dict[str, Any]]:
    """Report install/cache state without importing or loading a neural model."""
    return [
        {
            "provider": "sentence-transformers",
            "model": DEFAULT_SENTENCE_TRANSFORMER,
            "local": True,
            "available": importlib.util.find_spec("sentence_transformers") is not None,
            "dimension": None,
            "device": "cpu",
            "note": "Optional; cached model only, no download is initiated.",
        },
        {
            "provider": "ollama",
            "model": DEFAULT_OLLAMA_EMBEDDING,
            "local": True,
            "available": None,
            "dimension": None,
            "device": "local Ollama",
            "note": "Optional; requires the selected embedding model in the local Ollama installation.",
        },
        {
            "provider": "statistical_lsa",
            "model": "TF-IDF + TruncatedSVD",
            "local": True,
            "available": True,
            "dimension": None,
            "device": "CPU",
            "note": "Deterministic statistical latent space; explicitly not a neural embedding model.",
        },
    ]
