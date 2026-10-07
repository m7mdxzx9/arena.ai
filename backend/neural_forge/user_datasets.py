"""Secure, persistent user-dataset workspace.

Only tabular formats are accepted. Uploaded bytes are parsed as data and immediately
normalised to CSV; executable/pickle formats are never accepted. The API never exposes
storage paths.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from . import datalab
from .db import DB

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_XLSX_UNCOMPRESSED = 50 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 200
MAX_CELL_CHARS = 20_000
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".json", ".xlsx"}
_SAFE_NAME = re.compile(r"[^\w\-. ]+", re.UNICODE)


class DatasetUploadError(ValueError):
    """A safe validation error suitable for returning to a user."""

    def __init__(self, message: str, code: str = "invalid_dataset"):
        super().__init__(message)
        self.message = message
        self.code = code


def safe_display_name(filename: str) -> str:
    """Return a display-only basename; never use a client filename as a path."""
    raw = (filename or "dataset").replace("\\", "/").split("/")[-1]
    cleaned = _SAFE_NAME.sub("_", raw).strip(" ._")[:120]
    return cleaned or "dataset"


def _validate_xlsx_container(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            total = 0
            for item in archive.infolist():
                # XLSX is a ZIP. Reject traversal and expansion bombs before openpyxl sees it.
                parts = Path(item.filename).parts
                if item.filename.startswith(("/", "\\")) or ".." in parts:
                    raise DatasetUploadError("The spreadsheet contains an unsafe internal path.", "unsafe_archive")
                total += item.file_size
                if total > MAX_XLSX_UNCOMPRESSED:
                    raise DatasetUploadError("The expanded spreadsheet is too large.", "archive_too_large")
    except zipfile.BadZipFile as exc:
        raise DatasetUploadError("The XLSX file is not a valid spreadsheet.", "invalid_xlsx") from exc


def _read_frame(data: bytes, extension: str) -> pd.DataFrame:
    stream = io.BytesIO(data)
    try:
        if extension == ".csv":
            return pd.read_csv(stream, encoding="utf-8-sig", on_bad_lines="error")
        if extension == ".tsv":
            return pd.read_csv(stream, sep="\t", encoding="utf-8-sig", on_bad_lines="error")
        if extension == ".json":
            payload = json.loads(data.decode("utf-8-sig"))
            if isinstance(payload, dict) and "records" in payload:
                payload = payload["records"]
            if not isinstance(payload, (list, dict)):
                raise DatasetUploadError("JSON must contain an array of records or an object of columns.", "invalid_json_shape")
            return pd.DataFrame(payload)
        if extension == ".xlsx":
            _validate_xlsx_container(data)
            # read_excel reads cell values only. Macros are not an accepted format and no
            # spreadsheet formulas are evaluated by Neural Forge.
            return pd.read_excel(stream, engine="openpyxl")
    except DatasetUploadError:
        raise
    except UnicodeDecodeError as exc:
        raise DatasetUploadError("The file must use UTF-8 text encoding.", "invalid_encoding") from exc
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DatasetUploadError(f"The dataset could not be parsed: {str(exc)[:180]}", "parse_failed") from exc
    except ImportError as exc:
        raise DatasetUploadError("XLSX support is not installed. Install the optional spreadsheet dependency.", "xlsx_unavailable") from exc
    raise DatasetUploadError("Unsupported dataset format.", "unsupported_format")


def validate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate bounds and return a normalised copy plus transparent warnings."""
    if frame.empty or frame.shape[1] == 0:
        raise DatasetUploadError("The dataset has no rows or columns.", "empty_dataset")
    if len(frame) > MAX_ROWS:
        raise DatasetUploadError(f"The dataset has {len(frame):,} rows; the limit is {MAX_ROWS:,}.", "too_many_rows")
    if frame.shape[1] > MAX_COLUMNS:
        raise DatasetUploadError(f"The dataset has {frame.shape[1]} columns; the limit is {MAX_COLUMNS}.", "too_many_columns")

    out = frame.copy()
    names = [str(c).strip() for c in out.columns]
    if any(not c for c in names):
        raise DatasetUploadError("Every column must have a non-empty name.", "empty_column_name")
    if len(set(names)) != len(names):
        raise DatasetUploadError("Column names must be unique.", "duplicate_columns")
    out.columns = names

    warnings: list[str] = []
    formula_cells = 0
    for column in out.select_dtypes(include=["object", "string"]).columns:
        values = out[column].dropna().astype(str)
        if not values.empty and int(values.str.len().max()) > MAX_CELL_CHARS:
            raise DatasetUploadError(
                f"Column '{column}' contains a cell longer than {MAX_CELL_CHARS:,} characters.",
                "cell_too_large",
            )
        # Formula-looking values remain inert strings. Report them so exports can be
        # handled consciously rather than silently pretending they are ordinary text.
        formula_cells += int(values.str.match(r"^[=+@]").sum())
    if formula_cells:
        warnings.append(
            f"{formula_cells} cell(s) begin with a spreadsheet formula marker. "
            "They are stored as inert text and are never executed."
        )

    # Replace infinities with missing values so profiling and sklearn produce clear,
    # bounded behaviour rather than serialisation errors.
    numeric = out.select_dtypes(include=[np.number]).columns
    infinite = int(np.isinf(out[numeric].to_numpy(dtype=float, copy=True)).sum()) if len(numeric) else 0
    if infinite:
        out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
        warnings.append(f"{infinite} infinite numeric value(s) were converted to missing values.")
    return out, warnings


def parse_upload(file: BinaryIO | bytes, filename: str, declared_size: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse an untrusted upload without using its filename as a filesystem path."""
    display = safe_display_name(filename)
    extension = Path(display).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DatasetUploadError(f"Unsupported file type. Accepted types: {allowed}.", "unsupported_format")
    if isinstance(file, bytes):
        data = file
    else:
        data = file.read(MAX_UPLOAD_BYTES + 1)
    size = len(data)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        raise DatasetUploadError(f"The upload exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.", "file_too_large")
    if size > MAX_UPLOAD_BYTES:
        raise DatasetUploadError(f"The upload exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.", "file_too_large")
    if size == 0:
        raise DatasetUploadError("The uploaded file is empty.", "empty_file")
    frame, warnings = validate_frame(_read_frame(data, extension))
    return frame, {"original_name": display, "extension": extension[1:], "size_bytes": size, "warnings": warnings}


class DatasetStore:
    """Persistence façade for datasets owned by one player."""

    def __init__(self, db: DB, root: str | Path | None = None):
        if root is None:
            configured = os.environ.get("NEURAL_FORGE_UPLOAD_DIR")
            if configured:
                root = Path(configured)
            elif db.path == ":memory:":
                root = Path(tempfile.mkdtemp(prefix="neural_forge_uploads_"))
            else:
                root = Path(db.path).resolve().parent / "uploads"
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db

    def save(self, player_id: int, frame: pd.DataFrame, metadata: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        dataset_id = uuid.uuid4().hex
        stored_name = f"{dataset_id}.csv"
        path = self.root / stored_name
        # Atomic write avoids a database row pointing to a partial CSV after interruption.
        temp = self.root / f".{dataset_id}.tmp"
        frame.to_csv(temp, index=False)
        temp.replace(path)
        title = (name or Path(metadata["original_name"]).stem or "Dataset").strip()[:80]
        columns = [{"name": str(c), "dtype": str(frame[c].dtype), **datalab.infer_kind(frame[c])} for c in frame.columns]
        now = time.time()
        try:
            self.db.x(
                "INSERT INTO uploaded_datasets(id, player_id, name, original_name, format, stored_name, size_bytes, rows, columns_json, warnings_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    player_id,
                    title,
                    metadata["original_name"],
                    metadata["extension"],
                    stored_name,
                    int(metadata["size_bytes"]),
                    int(len(frame)),
                    json.dumps(columns, ensure_ascii=False),
                    json.dumps(metadata.get("warnings", []), ensure_ascii=False),
                    now,
                ),
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return self.get(player_id, dataset_id)

    def create_from_upload(self, player_id: int, file: BinaryIO | bytes, filename: str, name: str | None = None, declared_size: int | None = None) -> dict[str, Any]:
        frame, metadata = parse_upload(file, filename, declared_size)
        return self.save(player_id, frame, metadata, name)

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "original_name": row["original_name"],
            "format": row["format"],
            "size_bytes": row["size_bytes"],
            "rows": row["rows"],
            "columns": json.loads(row["columns_json"]),
            "warnings": json.loads(row["warnings_json"]),
            "created_at": row["created_at"],
            "source": "user-uploaded",
        }

    def list(self, player_id: int) -> list[dict[str, Any]]:
        rows = self.db.q("SELECT * FROM uploaded_datasets WHERE player_id=? ORDER BY created_at DESC", (player_id,))
        return [self._public(row) for row in rows]

    def get(self, player_id: int, dataset_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM uploaded_datasets WHERE id=? AND player_id=?", (dataset_id, player_id))
        if not row:
            raise KeyError(dataset_id)
        return self._public(row)

    def load(self, player_id: int, dataset_id: str) -> pd.DataFrame:
        row = self.db.one("SELECT stored_name FROM uploaded_datasets WHERE id=? AND player_id=?", (dataset_id, player_id))
        if not row:
            raise KeyError(dataset_id)
        path = (self.root / row["stored_name"]).resolve()
        if path.parent != self.root or not path.is_file():
            raise DatasetUploadError("The stored dataset file is unavailable.", "storage_missing")
        return pd.read_csv(path)

    def profile(self, player_id: int, dataset_id: str, target: str | None = None) -> dict[str, Any]:
        meta = self.get(player_id, dataset_id)
        frame = self.load(player_id, dataset_id)
        if target and target not in frame.columns:
            raise DatasetUploadError("The selected target column does not exist.", "unknown_target")
        return {"dataset": meta, "profile": datalab.profile(frame, target), "target": target}

    def delete(self, player_id: int, dataset_id: str) -> bool:
        row = self.db.one("SELECT stored_name FROM uploaded_datasets WHERE id=? AND player_id=?", (dataset_id, player_id))
        if not row:
            return False
        self.db.x("DELETE FROM uploaded_datasets WHERE id=? AND player_id=?", (dataset_id, player_id))
        path = (self.root / row["stored_name"]).resolve()
        if path.parent == self.root:
            path.unlink(missing_ok=True)
        return True
