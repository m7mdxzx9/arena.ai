"""Local-LLM provider abstraction and an Ollama implementation.

Core gameplay never imports or requires a model runtime. Provider failures are returned
as explicit availability errors; production paths never substitute fabricated output.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MAX_CHAT_CHARS = 40_000


class ProviderError(RuntimeError):
    def __init__(self, message: str, code: str = "provider_error", status: int = 503):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def health_check(self) -> dict[str, Any]: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def model_info(self, model: str) -> dict[str, Any]: ...
    def chat(self, model: str, messages: list[dict[str, str]], *, timeout: float | None = None) -> dict[str, Any]: ...
    def stream_chat(self, model: str, messages: list[dict[str, str]]) -> Iterator[dict[str, Any]]: ...
    def structured_generate(self, model: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]: ...
    def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]: ...


def _safe_base_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("OLLAMA_BASE_URL must be an http(s) URL without embedded credentials.")
    return value.rstrip("/")


def _validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ProviderError("At least one chat message is required.", "invalid_messages", 400)
    cleaned = []
    total = 0
    for message in messages[-30:]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ProviderError("Each message needs a valid role and text content.", "invalid_messages", 400)
        total += len(content)
        cleaned.append({"role": role, "content": content})
    if total > MAX_CHAT_CHARS:
        raise ProviderError(f"Chat context exceeds {MAX_CHAT_CHARS:,} characters.", "context_too_large", 413)
    return cleaned


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str | None = None, connect_timeout: float = 0.4, read_timeout: float = 120.0):
        self.base_url = _safe_base_url(base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
        self.timeout = httpx.Timeout(read_timeout, connect=connect_timeout)

    def _request(self, method: str, path: str, *, payload: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        try:
            response = httpx.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                timeout=timeout if timeout is not None else self.timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderError("Ollama did not respond before the timeout.", "ollama_timeout") from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                "Ollama is unavailable. Start Ollama locally and verify OLLAMA_BASE_URL.",
                "ollama_unavailable",
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = ""
            try:
                detail = str(exc.response.json().get("error", ""))[:300]
            except (ValueError, AttributeError):
                pass
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}" + (f": {detail}" if detail else "."),
                "ollama_http_error",
                502,
            ) from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise ProviderError("Ollama returned malformed JSON.", "ollama_invalid_response", 502) from exc

    def health_check(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            models = self.list_models()
            return {
                "provider": self.name,
                "reachable": True,
                "models_count": len(models),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except ProviderError as exc:
            return {
                "provider": self.name,
                "reachable": False,
                "models_count": 0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "error": exc.message,
                "code": exc.code,
            }

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/tags")
        models = data.get("models", []) if isinstance(data, dict) else []
        output = []
        for model in models:
            if not isinstance(model, dict) or not model.get("name"):
                continue
            details = model.get("details") if isinstance(model.get("details"), dict) else {}
            output.append(
                {
                    "provider": self.name,
                    "name": str(model["name"]),
                    "local": True,
                    "status": "available",
                    "size_bytes": model.get("size"),
                    "modified_at": model.get("modified_at"),
                    "digest": str(model.get("digest", ""))[:24] or None,
                    "format": details.get("format"),
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization": details.get("quantization_level"),
                    # /api/tags does not guarantee these capabilities; leave them
                    # unknown until /api/show reports them.
                    "capabilities": None,
                    "context_length": None,
                }
            )
        return output

    def model_info(self, model: str) -> dict[str, Any]:
        model = model.strip()[:160]
        if not model:
            raise ProviderError("A model name is required.", "model_required", 400)
        data = self._request("POST", "/api/show", payload={"model": model, "verbose": False})
        details = data.get("details", {}) if isinstance(data, dict) else {}
        model_info = data.get("model_info", {}) if isinstance(data, dict) else {}
        context = next(
            (value for key, value in model_info.items() if key.endswith(".context_length") and isinstance(value, int)),
            None,
        )
        capabilities = data.get("capabilities") if isinstance(data.get("capabilities"), list) else None
        return {
            "provider": self.name,
            "name": model,
            "local": True,
            "status": "available",
            "details": details,
            "capabilities": capabilities,
            "context_length": context,
        }

    def chat(self, model: str, messages: list[dict[str, str]], *, timeout: float | None = None) -> dict[str, Any]:
        cleaned = _validate_messages(messages)
        started = time.perf_counter()
        data = self._request(
            "POST",
            "/api/chat",
            payload={"model": model, "messages": cleaned, "stream": False},
            timeout=timeout,
        )
        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError("Ollama did not return a text message.", "ollama_invalid_response", 502)
        return {
            "provider": self.name,
            "model": data.get("model", model),
            "message": {"role": message.get("role", "assistant"), "content": content},
            "done": bool(data.get("done", True)),
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        }

    def stream_chat(self, model: str, messages: list[dict[str, str]]) -> Iterator[dict[str, Any]]:
        cleaned = _validate_messages(messages)
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": cleaned, "stream": True},
                timeout=self.timeout,
                follow_redirects=False,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError("Ollama emitted malformed stream data.", "ollama_invalid_response", 502) from exc
                    yield {
                        "content": str(event.get("message", {}).get("content", "")),
                        "done": bool(event.get("done", False)),
                        "model": event.get("model", model),
                    }
        except httpx.TimeoutException as exc:
            raise ProviderError("Ollama stream timed out.", "ollama_timeout") from exc
        except httpx.ConnectError as exc:
            raise ProviderError("Ollama is unavailable.", "ollama_unavailable") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Ollama stream failed with HTTP {exc.response.status_code}.", "ollama_http_error", 502) from exc

    def structured_generate(self, model: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        cleaned = _validate_messages(messages)
        data = self._request(
            "POST",
            "/api/chat",
            payload={"model": model, "messages": cleaned, "format": schema, "stream": False},
        )
        text = data.get("message", {}).get("content", "")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError("The model response was not valid JSON.", "invalid_structured_output", 502) from exc
        return {"provider": self.name, "model": data.get("model", model), "value": value}

    def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        if not inputs or len(inputs) > 128 or any(not isinstance(text, str) or len(text) > 20_000 for text in inputs):
            raise ProviderError("Embedding input must contain 1–128 bounded strings.", "invalid_embedding_input", 400)
        data = self._request("POST", "/api/embed", payload={"model": model, "input": inputs})
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise ProviderError("Ollama returned invalid embeddings.", "ollama_invalid_response", 502)
        return vectors


def get_provider(name: str = "ollama") -> LLMProvider:
    if name != "ollama":
        raise ProviderError(f"Unknown provider '{name}'.", "unknown_provider", 404)
    return OllamaProvider()
