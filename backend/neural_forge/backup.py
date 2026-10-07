"""Versioned, secret-redacted player backup and non-destructive restore."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from .db import DB

BACKUP_VERSION = 1
MAX_BACKUP_BYTES = 5 * 1024 * 1024
MAX_ROWS_PER_SECTION = 5_000
SAFE_SETTINGS = {"free_play", "show_code", "language", "default_model"}
SENSITIVE_FRAGMENTS = ("secret", "password", "api_key", "apikey", "access_token", "private_key", "credential")
FULL_BACKUP_FORMAT = "neural-forge-full-backup"
FULL_BACKUP_VERSION = 1
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 3 * 1024 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 5_000
MAX_ASSETS = 2_000


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


def _safe_file(root: Path, stored_name: str) -> Path:
    path = (root / stored_name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise ValueError("A guarded asset file is unavailable or has an invalid storage path.")
    return path


def _write_zip_file(archive: zipfile.ZipFile, archive_path: str, source: Path) -> tuple[int, str]:
    size = source.stat().st_size
    digest = hashlib.sha256()
    info = zipfile.ZipInfo(archive_path, date_time=time.localtime()[:6])
    info.compress_type = zipfile.ZIP_DEFLATED if size < 64 * 1024 * 1024 else zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    with source.open("rb") as incoming, archive.open(info, "w", force_zip64=True) as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            digest.update(chunk)
            outgoing.write(chunk)
    return size, digest.hexdigest()


def write_full_archive(db: DB, player_id: int, destination: str | Path, dataset_store: Any, document_store: Any, checkpoint_store: Any) -> dict[str, Any]:
    """Create a ZIP64 archive containing portable progress and guarded local assets.

    Uploaded tabular data and documents are already stored by the application in
    normalized CSV/text form. Checkpoints are copied as opaque bytes and are never
    deserialized here.
    """
    portable = export_player(db, player_id)
    portable["manifest"] = {
        "restore_mode": "merge",
        "excluded": ["agent execution traces"],
        "reason": "Execution traces are intentionally excluded; guarded datasets, documents, and generated checkpoints are embedded in this full backup.",
    }
    assets: list[dict[str, Any]] = []
    expanded = 0
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
            for index, row in enumerate(db.q("SELECT * FROM uploaded_datasets WHERE player_id=? ORDER BY created_at", (player_id,))):
                source = _safe_file(dataset_store.root, row["stored_name"])
                archive_path = f"assets/datasets/{index:06d}.csv"
                size, checksum = _write_zip_file(archive, archive_path, source)
                expanded += size
                assets.append({
                    "kind": "dataset", "path": archive_path, "sha256": checksum, "size_bytes": size,
                    "name": str(row["name"])[:80], "original_name": str(row["original_name"])[:255],
                    "source_format": str(row["format"])[:20], "created_at": row["created_at"],
                })
            for index, row in enumerate(db.q("SELECT * FROM documents WHERE player_id=? ORDER BY created_at", (player_id,))):
                source = _safe_file(document_store.root, row["stored_name"])
                archive_path = f"assets/documents/{index:06d}.txt"
                size, checksum = _write_zip_file(archive, archive_path, source)
                expanded += size
                assets.append({
                    "kind": "document", "path": archive_path, "sha256": checksum, "size_bytes": size,
                    "name": str(row["name"])[:100], "original_name": str(row["original_name"])[:255],
                    "source_format": str(row["format"])[:20], "source_size_bytes": int(row["size_bytes"]),
                    "created_at": row["created_at"],
                })
            for index, item in enumerate(checkpoint_store.list(player_id)):
                if not item.get("available"):
                    raise ValueError(f"Checkpoint '{item['name']}' is registered but its file is unavailable.")
                source = checkpoint_store.download_path(player_id, item["id"])
                archive_path = f"assets/checkpoints/{index:06d}.pt"
                size, checksum = _write_zip_file(archive, archive_path, source)
                expanded += size
                assets.append({
                    "kind": "checkpoint", "path": archive_path, "sha256": checksum, "size_bytes": size,
                    "name": str(item["name"])[:160], "checkpoint_kind": str(item["kind"])[:50],
                    "metadata": _redact(item.get("metadata", {})), "created_at": item["created_at"],
                })
            if len(assets) > MAX_ASSETS or expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ValueError("Full backup exceeds the guarded asset count or expanded-size limit.")
            manifest = {
                "format": FULL_BACKUP_FORMAT,
                "version": FULL_BACKUP_VERSION,
                "created_at": time.time(),
                "portable_backup": portable,
                "assets": assets,
                "integrity": "sha256-per-asset",
            }
            encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > MAX_BACKUP_BYTES * 2:
                raise ValueError("Full backup manifest exceeds the 10 MiB limit.")
            archive.writestr("backup.json", encoded)
        archive_size = destination.stat().st_size
        if archive_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Full backup archive exceeds the 2 GiB limit.")
        return {"assets": len(assets), "expanded_bytes": expanded, "archive_bytes": archive_size}
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _valid_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and "\\" not in name and all(part not in {"", ".", ".."} for part in path.parts)


def _verified_read(archive: zipfile.ZipFile, descriptor: dict[str, Any], limit: int) -> bytes:
    expected_size = int(descriptor.get("size_bytes", -1))
    expected_hash = str(descriptor.get("sha256", "")).lower()
    if expected_size < 0 or expected_size > limit or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("An archive asset has invalid size or integrity metadata.")
    digest = hashlib.sha256()
    output = bytearray()
    with archive.open(str(descriptor["path"]), "r") as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, limit + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
            digest.update(chunk)
            if len(output) > limit:
                raise ValueError("An archive asset exceeds its guarded size limit.")
    if len(output) != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError("Archive asset integrity verification failed.")
    return bytes(output)


def _verified_check(archive: zipfile.ZipFile, descriptor: dict[str, Any], limit: int) -> None:
    expected_size = int(descriptor.get("size_bytes", -1))
    expected_hash = str(descriptor.get("sha256", "")).lower()
    if expected_size < 0 or expected_size > limit or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("An archive asset has invalid size or integrity metadata.")
    digest = hashlib.sha256()
    size = 0
    with archive.open(str(descriptor["path"]), "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if size > limit:
                raise ValueError("An archive asset exceeds its guarded size limit.")
            digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_hash:
        raise ValueError("Archive asset integrity verification failed.")


def _verified_copy(archive: zipfile.ZipFile, descriptor: dict[str, Any], destination: Path, limit: int) -> None:
    expected_size = int(descriptor["size_bytes"])
    expected_hash = str(descriptor["sha256"]).lower()
    temporary = destination.with_suffix(".restore-tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(str(descriptor["path"]), "r") as incoming, temporary.open("wb") as outgoing:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                size += len(chunk)
                if size > limit:
                    raise ValueError("A checkpoint exceeds the guarded restore limit.")
                digest.update(chunk)
                outgoing.write(chunk)
        if size != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError("Checkpoint integrity verification failed during restore.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def restore_full_archive(db: DB, player_id: int, source: str | Path | BinaryIO, dataset_store: Any, document_store: Any, checkpoint_store: Any) -> dict[str, Any]:
    """Validate and merge a full backup without extracting archive-controlled paths."""
    from .document_rag import MAX_DOCUMENT_BYTES, extract_text
    from .user_datasets import MAX_UPLOAD_BYTES, parse_upload

    created_datasets: list[str] = []
    created_documents: list[str] = []
    created_checkpoints: list[str] = []
    with zipfile.ZipFile(source, "r", allowZip64=True) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if not infos or len(infos) > MAX_ARCHIVE_ENTRIES or len(set(names)) != len(names):
            raise ValueError("Backup archive has an invalid or excessive entry list.")
        expanded = 0
        for info in infos:
            if info.is_dir() or not _valid_archive_name(info.filename) or info.flag_bits & 0x1:
                raise ValueError("Backup archive contains an unsafe path, directory, or encrypted entry.")
            expanded += info.file_size
            if info.file_size > 512 * 1024 * 1024 or expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ValueError("Backup archive exceeds guarded expanded-size limits.")
        if "backup.json" not in names or archive.getinfo("backup.json").file_size > MAX_BACKUP_BYTES * 2:
            raise ValueError("Backup archive manifest is missing or too large.")
        try:
            manifest = json.loads(archive.read("backup.json"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Backup archive manifest is malformed.") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != FULL_BACKUP_FORMAT or manifest.get("version") != FULL_BACKUP_VERSION:
            raise ValueError("Unsupported NEURAL FORGE full-backup format or version.")
        portable = validate_backup(manifest.get("portable_backup"))
        assets = manifest.get("assets")
        if not isinstance(assets, list) or len(assets) > MAX_ASSETS or any(not isinstance(item, dict) for item in assets):
            raise ValueError("Backup archive asset manifest is invalid.")
        asset_paths: set[str] = set()
        allowed_kinds = {"dataset", "document", "checkpoint"}
        for item in assets:
            archive_path = item.get("path")
            if item.get("kind") not in allowed_kinds or not isinstance(archive_path, str) or not archive_path.startswith(f"assets/{item['kind']}s/") or archive_path in asset_paths:
                raise ValueError("Backup archive contains an invalid asset descriptor.")
            asset_paths.add(archive_path)
            if archive_path not in names:
                raise ValueError("Backup archive is missing a declared asset.")
        if set(names) != {"backup.json", *asset_paths}:
            raise ValueError("Backup archive contains undeclared files.")

        # Parse all normalized user content and verify every digest before making
        # profile or filesystem changes. Checkpoint bytes remain opaque.
        for item in assets:
            if item["kind"] == "dataset":
                data = _verified_read(archive, item, MAX_UPLOAD_BYTES)
                parse_upload(data, "restored.csv", len(data))
            elif item["kind"] == "document":
                data = _verified_read(archive, item, MAX_DOCUMENT_BYTES)
                extract_text(data, "restored.txt")
            else:
                _verified_check(archive, item, 512 * 1024 * 1024)

        restored = restore_player(db, player_id, portable)
        try:
            asset_counts = {"datasets": 0, "documents": 0, "checkpoints": 0}
            for item in assets:
                title = str(item.get("name", "Restored asset")).strip()[:160] or "Restored asset"
                if item["kind"] == "dataset":
                    data = _verified_read(archive, item, MAX_UPLOAD_BYTES)
                    frame, metadata = parse_upload(data, "restored.csv", len(data))
                    original_name = Path(str(item.get("original_name", "restored.csv"))).name[:255] or "restored.csv"
                    metadata.update({"original_name": original_name, "extension": str(item.get("source_format", "csv"))[:20] or "csv"})
                    saved = dataset_store.save(player_id, frame, metadata, title)
                    created_datasets.append(saved["id"])
                    asset_counts["datasets"] += 1
                elif item["kind"] == "document":
                    data = _verified_read(archive, item, MAX_DOCUMENT_BYTES)
                    saved = document_store.add(player_id, data, "restored.txt", title)
                    created_documents.append(saved["id"])
                    original_name = Path(str(item.get("original_name", "restored.txt"))).name[:255] or "restored.txt"
                    source_format = str(item.get("source_format", "txt"))[:20] or "txt"
                    source_size = max(0, min(MAX_DOCUMENT_BYTES, int(item.get("source_size_bytes", len(data)))))
                    db.x("UPDATE documents SET original_name=?, format=?, size_bytes=? WHERE id=? AND player_id=?", (original_name, source_format, source_size, saved["id"], player_id))
                    asset_counts["documents"] += 1
                else:
                    checkpoint_id, checkpoint_path = checkpoint_store.allocate(player_id)
                    _verified_copy(archive, item, checkpoint_path, 512 * 1024 * 1024)
                    run_id = db.save_run(player_id, "checkpoint_restore", {"source": "full_backup"}, {}, {"checkpoint_name": title}, name=f"Restored checkpoint: {title}")
                    saved = checkpoint_store.register(
                        player_id, checkpoint_id, checkpoint_path, run_id,
                        str(item.get("checkpoint_kind", "pytorch"))[:50] or "pytorch",
                        title, item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
                    )
                    created_checkpoints.append(saved["id"])
                    asset_counts["checkpoints"] += 1
            restored["assets"] = asset_counts
            restored["archive_version"] = FULL_BACKUP_VERSION
            return restored
        except Exception as exc:
            for asset_id in reversed(created_checkpoints):
                checkpoint_store.delete(player_id, asset_id)
            for asset_id in reversed(created_documents):
                document_store.delete(player_id, asset_id)
            for asset_id in reversed(created_datasets):
                dataset_store.delete(player_id, asset_id)
            raise ValueError("Profile merge completed, but guarded asset restoration failed and created assets were rolled back.") from exc
