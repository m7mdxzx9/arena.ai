"""Truthful, non-sensitive runtime capability reporting."""
from __future__ import annotations

import importlib.util
import time
from threading import Lock
from typing import Any

from . import torch_engine
from .curriculum import CONCEPTS
from .curriculum.translations_ar import AR_CONCEPTS
from .embeddings import provider_capabilities
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
                "learning_simulator_details": {
                    "source_language": "en",
                    "free_form_outputs_language": "en",
                    "benchmark_questions_localized": True,
                    "neural_embeddings": False,
                    "representations": ["tfidf", "lsa", "hash32"],
                    "lexical_retrieval": "bm25",
                    "reranker": "lexical_overlap_heuristic",
                },
                "personal_document_formats": ["pdf", "txt", "md", "docx"],
                "embedding_providers": provider_capabilities(),
                "rerankers": [
                    {"id": "none", "kind": "disabled", "available": True, "local_only": True},
                    {"id": "lexical_overlap", "kind": "heuristic_not_neural", "available": True, "local_only": True},
                    {"id": "cross_encoder", "kind": "optional_local_neural", "dependency_installed": importlib.util.find_spec("sentence_transformers") is not None, "model_cache_required": True, "automatic_download": False},
                ],
                "answer_modes": ["offline_extractive", "explicit_local_ollama"],
            },
            "tutor": {
                "available": True,
                "curated_offline": True,
                "player_state_personalization": True,
                "local_ollama_optional": True,
                "raw_questions_and_answers_persisted": False,
                "feedback_awards_mastery": False,
            },
            "learning_missions": {
                "evidence_checked": ["rag_compare", "rag_grounding", "tutor_hint_ladder"],
                "retrieval_boss_personal_evidence_phase": True,
            },
            "agents": {
                "security_simulator": True,
                "real_agent_lab": True,
            },
            "localization": {
                "languages": ["en", "ar"],
                "arabic_rtl": True,
                "coverage": "partial",
                "curriculum_concepts": len(CONCEPTS),
                "reviewed_arabic_concepts": len(AR_CONCEPTS),
                "simulator_benchmark_questions_translated": True,
                "simulator_source_language": "en",
            },
            "datasets": {"user_upload": True, "formats": ["csv", "tsv", "json", "xlsx"]},
        }
        _CACHE = (now, result)
        return result
