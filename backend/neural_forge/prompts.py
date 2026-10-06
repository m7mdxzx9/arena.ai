"""Persistent prompt versioning with explicit variables and real provider execution."""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from .db import DB
from .llm import LLMProvider, get_provider

VARIABLE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def _version(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"], "prompt_id": row["prompt_id"], "version": row["version"],
        "system": row["system_text"], "user": row["user_text"], "variables": json.loads(row["variables"]),
        "change_note": row["change_note"], "created_at": row["created_at"],
    }


def create(db: DB, player_id: int, name: str, system: str, user: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    name, system, user = name.strip()[:100], system.strip()[:10_000], user.strip()[:10_000]
    if not name or not user:
        raise ValueError("Prompt name and user section are required.")
    prompt_id = uuid.uuid4().hex
    now = time.time()
    db.x("INSERT INTO prompts(id, player_id, name, created_at, updated_at) VALUES (?,?,?,?,?)", (prompt_id, player_id, name, now, now))
    db.x(
        "INSERT INTO prompt_versions(prompt_id, player_id, version, system_text, user_text, variables, change_note, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (prompt_id, player_id, 1, system, user, json.dumps(variables or {}, ensure_ascii=False), "Initial version", now),
    )
    return get(db, player_id, prompt_id)


def add_version(db: DB, player_id: int, prompt_id: str, system: str, user: str, variables: dict[str, Any] | None = None, change_note: str = "") -> dict[str, Any]:
    if not db.one("SELECT 1 FROM prompts WHERE id=? AND player_id=?", (prompt_id, player_id)):
        raise KeyError(prompt_id)
    row = db.one("SELECT MAX(version) latest FROM prompt_versions WHERE prompt_id=? AND player_id=?", (prompt_id, player_id))
    number = int(row["latest"] or 0) + 1
    now = time.time()
    db.x(
        "INSERT INTO prompt_versions(prompt_id, player_id, version, system_text, user_text, variables, change_note, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (prompt_id, player_id, number, system.strip()[:10_000], user.strip()[:10_000], json.dumps(variables or {}, ensure_ascii=False), change_note.strip()[:500], now),
    )
    db.x("UPDATE prompts SET updated_at=? WHERE id=? AND player_id=?", (now, prompt_id, player_id))
    return get(db, player_id, prompt_id)


def list_prompts(db: DB, player_id: int) -> list[dict[str, Any]]:
    rows = db.q(
        "SELECT p.*, MAX(v.version) latest_version, COUNT(v.id) versions FROM prompts p JOIN prompt_versions v ON v.prompt_id=p.id "
        "WHERE p.player_id=? GROUP BY p.id ORDER BY p.updated_at DESC",
        (player_id,),
    )
    return [dict(row) for row in rows]


def get(db: DB, player_id: int, prompt_id: str) -> dict[str, Any]:
    prompt = db.one("SELECT * FROM prompts WHERE id=? AND player_id=?", (prompt_id, player_id))
    if not prompt:
        raise KeyError(prompt_id)
    versions = [_version(row) for row in db.q("SELECT * FROM prompt_versions WHERE prompt_id=? AND player_id=? ORDER BY version", (prompt_id, player_id))]
    return {**dict(prompt), "versions": versions}


def render(template: str, variables: dict[str, Any]) -> str:
    required = set(VARIABLE_RE.findall(template))
    missing = sorted(required - set(variables))
    if missing:
        raise ValueError(f"Missing prompt variables: {', '.join(missing)}")
    values: dict[str, str] = {}
    for name in required:
        value = variables[name]
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Variable '{name}' must be a string, number, or boolean.")
        text = str(value)
        if len(text) > 5_000:
            raise ValueError(f"Variable '{name}' is too long.")
        values[name] = text
    return VARIABLE_RE.sub(lambda match: values[match.group(1)], template)


def execute(
    db: DB,
    player_id: int,
    prompt_id: str,
    version: int,
    variables: dict[str, Any],
    model: str,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    row = db.one(
        "SELECT * FROM prompt_versions WHERE prompt_id=? AND player_id=? AND version=?",
        (prompt_id, player_id, version),
    )
    if not row:
        raise KeyError(prompt_id)
    system_text = render(row["system_text"], variables)
    user_text = render(row["user_text"], variables)
    provider = provider or get_provider("ollama")
    response = provider.chat(model, [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}])
    return {
        "prompt_id": prompt_id,
        "version": version,
        "model": response["model"],
        "provider": response["provider"],
        "output": response["message"]["content"],
        "duration_ms": response["duration_ms"],
        "rendered": {"system": system_text, "user": user_text},
    }
