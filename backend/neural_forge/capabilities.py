"""Truthful, non-sensitive runtime capability reporting."""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

from . import torch_engine
from .llm import OllamaProvider

_CACHE: tuple[float, dict[str, Any]] | None = None
_LOCK = Lock()
CACHE_SECONDS = 5.0


def system_capabilities(force: bool = False) -> dict[str, Any]:
    """Return only product capabilities safe to expose to a browser."""
    global _CACHE
    now = time.monotonic()
    with _LOCK:
        if not force and _CACHE and now - _CACHE[0] < CACHE_SECONDS:
            return _CACHE[1]
        torch_status = torch_engine.availability()
        ollama = OllamaProvider(connect_timeout=0.2, read_timeout=0.5).health_check()
        result = {
            "pytorch": {
                "installed": torch_status["installed"],
                "version": torch_status["version"],
                "cuda_available": torch_status["cuda_available"],
                "cuda_devices": torch_status["cuda_devices"],
            },
            "ollama": {
                "reachable": ollama["reachable"],
                "configured_local_models": ollama["models_count"],
                "status": "available" if ollama["reachable"] else "unavailable",
            },
            "rag": {
                "learning_simulator": True,
                "built_in_retrieval": True,
                "personal_documents": True,
            },
            "agents": {
                "security_simulator": True,
                "real_agent_lab": True,
            },
            "localization": {"languages": ["en", "ar"], "arabic_rtl": True},
            "datasets": {"user_upload": True, "formats": ["csv", "tsv", "json", "xlsx"]},
        }
        _CACHE = (now, result)
        return result
