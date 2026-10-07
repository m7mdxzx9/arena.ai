"""Registry and guarded storage for checkpoints produced by NEURAL FORGE.

Only files created by the local training engine can be registered. This module never
loads a checkpoint, so PyTorch/pickle deserialization is not exposed to uploads.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .db import DB


class CheckpointError(ValueError):
    def __init__(self, message: str, code: str = "checkpoint_error"):
        super().__init__(message)
        self.message = message
        self.code = code


class CheckpointStore:
    def __init__(self, db: DB, root: str | Path | None = None):
        if root is None:
            configured = os.environ.get("NEURAL_FORGE_CHECKPOINT_DIR")
            base = Path(db.path).resolve().parent if db.path != ":memory:" else Path(os.environ.get("TMPDIR", "/tmp")) / "neural-forge-memory"
            root = Path(configured).expanduser() if configured else base / "checkpoints"
        self.db = db
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _player_root(self, player_id: int) -> Path:
        root = (self.root / str(player_id)).resolve()
        if self.root not in root.parents:
            raise CheckpointError("Invalid checkpoint storage root.", "invalid_path")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def allocate(self, player_id: int) -> tuple[str, Path]:
        checkpoint_id = uuid.uuid4().hex
        return checkpoint_id, self._player_root(player_id) / f"{checkpoint_id}.pt"

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def register(
        self,
        player_id: int,
        checkpoint_id: str,
        path: Path,
        run_id: int,
        kind: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected = (self._player_root(player_id) / f"{checkpoint_id}.pt").resolve()
        resolved = path.resolve()
        if resolved != expected or not resolved.is_file():
            raise CheckpointError("Training did not produce the expected checkpoint file.", "checkpoint_missing")
        if resolved.stat().st_size > 512 * 1024 * 1024:
            self._delete_files(resolved)
            raise CheckpointError("Generated checkpoint exceeds the 512 MiB storage limit.", "checkpoint_too_large")
        sidecar = resolved.with_suffix(".json")
        sidecar_metadata: dict[str, Any] = {}
        if sidecar.is_file():
            try:
                value = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    sidecar_metadata = value
            except (OSError, json.JSONDecodeError):
                sidecar_metadata = {"warning": "Checkpoint metadata sidecar could not be parsed."}
        combined = {**sidecar_metadata, **(metadata or {})}
        now = time.time()
        display_name = (name or f"{kind.upper()} run #{run_id}").strip()[:160] or f"{kind.upper()} run #{run_id}"
        self.db.x(
            "INSERT INTO model_checkpoints(id, player_id, run_id, kind, name, stored_name, metadata_json, size_bytes, sha256, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (checkpoint_id, player_id, run_id, kind[:50], display_name, resolved.name, json.dumps(combined, ensure_ascii=False), resolved.stat().st_size, self._hash(resolved), now),
        )
        return self.get(player_id, checkpoint_id)

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"], "run_id": row["run_id"], "kind": row["kind"], "name": row["name"],
            "size_bytes": row["size_bytes"], "sha256": row["sha256"], "created_at": row["created_at"],
            "format": "pytorch-state-dict", "metadata": json.loads(row["metadata_json"]),
        }

    def list(self, player_id: int) -> list[dict[str, Any]]:
        rows = self.db.q("SELECT * FROM model_checkpoints WHERE player_id=? ORDER BY created_at DESC", (player_id,))
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self._public(row)
            item["available"] = self._path_from_row(player_id, row).is_file()
            result.append(item)
        return result

    def _row(self, player_id: int, checkpoint_id: str) -> Any:
        row = self.db.one("SELECT * FROM model_checkpoints WHERE id=? AND player_id=?", (checkpoint_id, player_id))
        if not row:
            raise KeyError(checkpoint_id)
        return row

    def get(self, player_id: int, checkpoint_id: str) -> dict[str, Any]:
        row = self._row(player_id, checkpoint_id)
        item = self._public(row)
        item["available"] = self._path_from_row(player_id, row).is_file()
        return item

    def _path_from_row(self, player_id: int, row: Any) -> Path:
        root = self._player_root(player_id)
        path = (root / row["stored_name"]).resolve()
        if root not in path.parents or path.suffix != ".pt":
            raise CheckpointError("Checkpoint path failed validation.", "invalid_path")
        return path

    def download_path(self, player_id: int, checkpoint_id: str) -> Path:
        row = self._row(player_id, checkpoint_id)
        path = self._path_from_row(player_id, row)
        if not path.is_file():
            raise CheckpointError("Checkpoint file is missing from local storage.", "checkpoint_missing")
        return path

    def rename(self, player_id: int, checkpoint_id: str, name: str) -> dict[str, Any]:
        self._row(player_id, checkpoint_id)
        clean = name.strip()[:160]
        if not clean:
            raise CheckpointError("Checkpoint name cannot be empty.", "invalid_name")
        self.db.x("UPDATE model_checkpoints SET name=? WHERE id=? AND player_id=?", (clean, checkpoint_id, player_id))
        return self.get(player_id, checkpoint_id)

    @staticmethod
    def _delete_files(path: Path) -> None:
        for candidate in (path, path.with_suffix(".json")):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:
                raise CheckpointError("Could not remove checkpoint from local storage.", "delete_failed") from exc

    def discard(self, player_id: int, checkpoint_id: str, path: Path) -> None:
        """Compensate a failed training/registration attempt without trusting paths."""
        expected = (self._player_root(player_id) / f"{checkpoint_id}.pt").resolve()
        if path.resolve() != expected:
            raise CheckpointError("Checkpoint cleanup path failed validation.", "invalid_path")
        self._delete_files(expected)
        self.db.x("DELETE FROM model_checkpoints WHERE id=? AND player_id=?", (checkpoint_id, player_id))

    def delete(self, player_id: int, checkpoint_id: str) -> dict[str, Any]:
        row = self._row(player_id, checkpoint_id)
        path = self._path_from_row(player_id, row)
        self._delete_files(path)
        self.db.x("DELETE FROM model_checkpoints WHERE id=? AND player_id=?", (checkpoint_id, player_id))
        return {"deleted": True, "id": checkpoint_id}
