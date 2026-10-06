"""Versioned, secret-redacted player backup and non-destructive restore."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from .db import DB

BACKUP_VERSION = 1
MAX_BACKUP_BYTES = 5 * 1024 * 1024
MAX_ROWS_PER_SECTION = 5_000
SAFE_SETTINGS = {"free_play", "show_code", "language", "default_model"}
SENSITIVE_FRAGMENTS = ("secret", "password", "api_key", "apikey", "access_token", "private_key", "credential")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items() if not any(part in str(key).casefold() for part in SENSITIVE_FRAGMENTS)}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _rows(db: DB, table: str, player_id: int, *, omit: set[str] = set(), json_fields: set[str] = set()) -> list[dict[str, Any]]:
    result = []
    for row in db.q(f"SELECT * FROM {table} WHERE player_id=? ORDER BY rowid LIMIT ?", (player_id, MAX_ROWS_PER_SECTION)):
        item = {key: row[key] for key in row.keys() if key not in omit | {"player_id"}}
        for field in json_fields:
            if field in item:
                try:
                    item[field] = json.loads(item[field])
                except (TypeError, json.JSONDecodeError):
                    item[field] = None
        result.append(_redact(item))
    return result


def export_player(db: DB, player_id: int) -> dict[str, Any]:
    player = db.player(player_id)
    if not player:
        raise KeyError(player_id)
    settings = {key: value for key, value in player["settings"].items() if key in SAFE_SETTINGS}
    data = {
        "format": "neural-forge-player-backup",
        "version": BACKUP_VERSION,
        "exported_at": time.time(),
        "profile": {"name": player["name"], "mode": player["mode"], "xp": player["xp"], "settings": settings},
        "mastery": _rows(db, "mastery", player_id, json_fields={"state"}),
        "attempts": _rows(db, "attempts", player_id, omit={"id"}),
        "progress": _rows(db, "progress", player_id),
        "achievements": _rows(db, "achievements", player_id),
        "reflections": _rows(db, "reflections", player_id, omit={"id"}),
        "runs": _rows(db, "runs", player_id, omit={"id"}, json_fields={"config", "result", "summary"}),
        "mistakes": _rows(db, "mistakes", player_id, omit={"id", "run_id"}),
        "prompts": _rows(db, "prompts", player_id),
        "prompt_versions": _rows(db, "prompt_versions", player_id, omit={"id"}, json_fields={"variables"}),
        "evaluation_datasets": _rows(db, "evaluation_datasets", player_id, json_fields={"cases_json"}),
        "agent_configurations": _rows(db, "agent_configurations", player_id, json_fields={"tools_json", "permissions_json", "memory_json"}),
        "portfolio_projects": _rows(db, "portfolio_projects", player_id, json_fields={"content"}),
        "manifest": {
            "restore_mode": "merge",
            "excluded": ["uploaded dataset files", "personal documents and chunks", "model checkpoints", "agent execution traces"],
            "reason": "Binary/private source files use separate guarded storage and are not embedded in portable JSON backups.",
        },
    }
    if len(json.dumps(data, ensure_ascii=False).encode("utf-8")) > MAX_BACKUP_BYTES:
        raise ValueError("This profile exceeds the 5 MiB portable-backup limit. Remove large experiment history and try again.")
    return data


def validate_backup(data: Any) -> dict[str, Any]:
    try:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Backup must be valid JSON data.") from exc
    if len(encoded) > MAX_BACKUP_BYTES:
        raise ValueError("Backup exceeds the 5 MiB limit.")
    if not isinstance(data, dict) or data.get("format") != "neural-forge-player-backup":
        raise ValueError("This is not a NEURAL FORGE player backup.")
    if data.get("version") != BACKUP_VERSION:
        raise ValueError(f"Unsupported backup version. Expected version {BACKUP_VERSION}.")
    if not isinstance(data.get("profile"), dict):
        raise ValueError("Backup profile is missing.")
    sections = ("mastery", "attempts", "progress", "achievements", "reflections", "runs", "mistakes", "prompts", "prompt_versions", "evaluation_datasets", "agent_configurations", "portfolio_projects")
    for section in sections:
        values = data.get(section, [])
        if not isinstance(values, list) or len(values) > MAX_ROWS_PER_SECTION or any(not isinstance(item, dict) for item in values):
            raise ValueError(f"Backup section '{section}' is invalid or too large.")
    return data


def restore_player(db: DB, player_id: int, raw: Any) -> dict[str, Any]:
    data = validate_backup(raw)
    if not db.player(player_id):
        raise KeyError(player_id)
    counts: dict[str, int] = {}
    now = time.time()
    prompt_map: dict[str, str] = {}
    with db.lock:
        connection = db.conn
        try:
            connection.execute("BEGIN")
            profile = data["profile"]
            current = db.player(player_id) or {}
            incoming_settings = profile.get("settings", {}) if isinstance(profile.get("settings"), dict) else {}
            settings = {**current.get("settings", {}), **{key: value for key, value in incoming_settings.items() if key in SAFE_SETTINGS}}
            mode = max(1, min(4, int(profile.get("mode", current.get("mode", 1)))))
            xp = max(int(current.get("xp", 0)), max(0, int(profile.get("xp", 0))))
            connection.execute("UPDATE players SET mode=?, xp=?, settings=? WHERE id=?", (mode, xp, json.dumps(settings), player_id))

            for item in data.get("mastery", []):
                cid, state = str(item.get("concept_id", ""))[:100], item.get("state")
                if cid and isinstance(state, dict):
                    connection.execute("INSERT INTO mastery(player_id, concept_id, state) VALUES (?,?,?) ON CONFLICT(player_id,concept_id) DO UPDATE SET state=excluded.state", (player_id, cid, json.dumps(state)))
                    counts["mastery"] = counts.get("mastery", 0) + 1

            simple_specs = {
                "attempts": ("concept_id,kind,item_key,difficulty,correct,hint_used,p_before,p_after,ts", ("concept_id", "kind", "item_key", "difficulty", "correct", "hint_used", "p_before", "p_after", "ts")),
                "reflections": ("concept_id,text,ts", ("concept_id", "text", "ts")),
                "mistakes": ("concept,category,mission_id,mistake_type,player_action,correct_principle,explanation,example,created_at,review_count,resolved,due_at", ("concept", "category", "mission_id", "mistake_type", "player_action", "correct_principle", "explanation", "example", "created_at", "review_count", "resolved", "due_at")),
            }
            for section, (columns, fields) in simple_specs.items():
                placeholders = ",".join("?" for _ in fields)
                for item in data.get(section, []):
                    values = tuple(item.get(field) for field in fields)
                    connection.execute(f"INSERT INTO {section}(player_id,{columns}) VALUES (?,{placeholders})", (player_id, *values))
                    counts[section] = counts.get(section, 0) + 1

            for item in data.get("progress", []):
                kind, item_id = str(item.get("kind", ""))[:50], str(item.get("item_id", ""))[:120]
                if kind and item_id:
                    connection.execute("INSERT INTO progress(player_id,kind,item_id,state,completed_at) VALUES (?,?,?,?,?) ON CONFLICT(player_id,kind,item_id) DO UPDATE SET state=excluded.state,completed_at=COALESCE(excluded.completed_at,progress.completed_at)", (player_id, kind, item_id, str(item.get("state", "available"))[:50], item.get("completed_at")))
                    counts["progress"] = counts.get("progress", 0) + 1
            for item in data.get("achievements", []):
                ach_id = str(item.get("ach_id", ""))[:120]
                if ach_id:
                    connection.execute("INSERT OR IGNORE INTO achievements(player_id,ach_id,ts) VALUES (?,?,?)", (player_id, ach_id, item.get("ts", now)))
                    counts["achievements"] = counts.get("achievements", 0) + 1
            for item in data.get("runs", []):
                connection.execute("INSERT INTO runs(player_id,kind,name,config,result,summary,notes,context,ts) VALUES (?,?,?,?,?,?,?,?,?)", (player_id, str(item.get("kind", "restored"))[:100], str(item.get("name", ""))[:200] or None, json.dumps(item.get("config", {})), json.dumps(item.get("result", {})), json.dumps(item.get("summary", {})), str(item.get("notes", ""))[:10_000], str(item.get("context", ""))[:100] or None, item.get("ts", now)))
                counts["runs"] = counts.get("runs", 0) + 1

            versions_by_prompt: dict[str, list[dict[str, Any]]] = {}
            for version in data.get("prompt_versions", []):
                versions_by_prompt.setdefault(str(version.get("prompt_id", "")), []).append(version)
            for item in data.get("prompts", []):
                old_id, new_id = str(item.get("id", "")), uuid.uuid4().hex
                prompt_map[old_id] = new_id
                connection.execute("INSERT INTO prompts(id,player_id,name,created_at,updated_at) VALUES (?,?,?,?,?)", (new_id, player_id, str(item.get("name", "Restored prompt"))[:100], item.get("created_at", now), now))
                for version in versions_by_prompt.get(old_id, []):
                    connection.execute("INSERT INTO prompt_versions(prompt_id,player_id,version,system_text,user_text,variables,change_note,created_at) VALUES (?,?,?,?,?,?,?,?)", (new_id, player_id, int(version.get("version", 1)), str(version.get("system_text", ""))[:10_000], str(version.get("user_text", ""))[:10_000], json.dumps(version.get("variables", {})), str(version.get("change_note", ""))[:500], version.get("created_at", now)))
                counts["prompts"] = counts.get("prompts", 0) + 1

            for item in data.get("evaluation_datasets", []):
                connection.execute("INSERT INTO evaluation_datasets(id,player_id,name,cases_json,created_at,updated_at) VALUES (?,?,?,?,?,?)", (uuid.uuid4().hex, player_id, str(item.get("name", "Restored evaluation"))[:100], json.dumps(item.get("cases_json", [])), item.get("created_at", now), now))
                counts["evaluation_datasets"] = counts.get("evaluation_datasets", 0) + 1
            for item in data.get("agent_configurations", []):
                connection.execute("INSERT INTO agent_configurations(id,player_id,name,model,system_prompt,tools_json,permissions_json,memory_json,max_steps,timeout_seconds,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, player_id, str(item.get("name", "Restored agent"))[:100], str(item.get("model", ""))[:160], str(item.get("system_prompt", ""))[:4_000], json.dumps(item.get("tools_json", [])), json.dumps(item.get("permissions_json", [])), json.dumps(item.get("memory_json", [])), max(1, min(12, int(item.get("max_steps", 6)))), max(5, min(180, float(item.get("timeout_seconds", 60)))), item.get("created_at", now), now))
                counts["agent_configurations"] = counts.get("agent_configurations", 0) + 1
            for item in data.get("portfolio_projects", []):
                content = item.get("content", {})
                if isinstance(content, dict):
                    title = str(content.get("title") or item.get("title") or "Restored project")[:160]
                    connection.execute("INSERT INTO portfolio_projects(id,player_id,title,content,created_at,updated_at) VALUES (?,?,?,?,?,?)", (uuid.uuid4().hex, player_id, title, json.dumps(content, ensure_ascii=False), item.get("created_at", now), now))
                    counts["portfolio_projects"] = counts.get("portfolio_projects", 0) + 1
            connection.commit()
        except (ValueError, TypeError, sqlite3.Error) as exc:
            connection.rollback()
            raise ValueError("Backup contents failed validation; no changes were restored.") from exc
    return {"restored": True, "mode": "merge", "counts": counts, "warnings": data.get("manifest", {}).get("excluded", [])}
