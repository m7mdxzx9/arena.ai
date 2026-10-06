"""SQLite persistence (stdlib only). One file, JSON columns for flexible payloads."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("NEURAL_FORGE_DB", Path(__file__).resolve().parent.parent / "neural_forge_data" / "neural_forge.sqlite3"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  mode INTEGER NOT NULL DEFAULT 1,
  xp INTEGER NOT NULL DEFAULT 0,
  settings TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mastery (
  player_id INTEGER NOT NULL,
  concept_id TEXT NOT NULL,
  state TEXT NOT NULL,
  PRIMARY KEY (player_id, concept_id)
);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL,
  concept_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  item_key TEXT,
  difficulty INTEGER,
  correct INTEGER NOT NULL,
  hint_used INTEGER NOT NULL DEFAULT 0,
  p_before REAL, p_after REAL,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_player ON attempts(player_id, concept_id);
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  name TEXT,
  config TEXT NOT NULL,
  result TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '{}',
  notes TEXT NOT NULL DEFAULT '',
  context TEXT,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_player ON runs(player_id, kind);
CREATE TABLE IF NOT EXISTS progress (
  player_id INTEGER NOT NULL,
  kind TEXT NOT NULL,          -- mission | boss | challenge | exercise | prediction
  item_id TEXT NOT NULL,
  state TEXT NOT NULL,
  completed_at REAL,
  PRIMARY KEY (player_id, kind, item_id)
);
CREATE TABLE IF NOT EXISTS achievements (
  player_id INTEGER NOT NULL,
  ach_id TEXT NOT NULL,
  ts REAL NOT NULL,
  PRIMARY KEY (player_id, ach_id)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL,
  xp INTEGER NOT NULL,
  reason TEXT NOT NULL,
  ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reflections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  player_id INTEGER NOT NULL,
  concept_id TEXT NOT NULL,
  text TEXT NOT NULL,
  ts REAL NOT NULL
);
"""


class DB:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or DEFAULT_PATH)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
            self.conn.commit()

    # -- helpers
    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()):
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def x(self, sql: str, args: tuple = ()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # -- players
    def create_player(self, name: str, mode: int = 1) -> int:
        return self.x("INSERT INTO players(name, mode, xp, settings, created_at) VALUES (?,?,?,?,?)",
                      (name, mode, 0, json.dumps({"free_play": False, "show_code": mode >= 4}), time.time()))

    def player(self, pid: int) -> dict | None:
        r = self.one("SELECT * FROM players WHERE id=?", (pid,))
        if not r:
            return None
        d = dict(r)
        d["settings"] = json.loads(d["settings"])
        return d

    def players(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT id, name, mode, xp, created_at FROM players ORDER BY id")]

    def update_player(self, pid: int, **fields):
        bad = set(fields) - {"name", "mode", "xp", "settings"}
        if bad:
            raise ValueError(f"Cannot update player fields {bad}")
        if "settings" in fields:
            fields["settings"] = json.dumps(fields["settings"])
        sets = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE players SET {sets} WHERE id=?", (*fields.values(), pid))

    def add_xp(self, pid: int, amount: int, reason: str):
        if amount:
            self.x("UPDATE players SET xp = MAX(0, xp + ?) WHERE id=?", (amount, pid))
            self.x("INSERT INTO events(player_id, xp, reason, ts) VALUES (?,?,?,?)", (pid, amount, reason, time.time()))

    # -- mastery
    def mastery(self, pid: int) -> dict[str, dict]:
        return {r["concept_id"]: json.loads(r["state"]) for r in self.q("SELECT concept_id, state FROM mastery WHERE player_id=?", (pid,))}

    def set_mastery(self, pid: int, cid: str, state: dict):
        self.x("INSERT INTO mastery(player_id, concept_id, state) VALUES (?,?,?) ON CONFLICT(player_id, concept_id) DO UPDATE SET state=excluded.state",
               (pid, cid, json.dumps(state)))

    def log_attempt(self, pid, cid, kind, item_key, difficulty, correct, hint, p_before, p_after):
        self.x("INSERT INTO attempts(player_id, concept_id, kind, item_key, difficulty, correct, hint_used, p_before, p_after, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (pid, cid, kind, item_key, difficulty, int(correct), int(hint), p_before, p_after, time.time()))

    def seen_items(self, pid: int, cid: str) -> list[str]:
        return [r["item_key"] for r in self.q("SELECT item_key FROM attempts WHERE player_id=? AND concept_id=? ORDER BY id DESC LIMIT 50", (pid, cid))]

    # -- runs
    def save_run(self, pid, kind, config, result, summary, name=None, context=None) -> int:
        return self.x("INSERT INTO runs(player_id, kind, name, config, result, summary, context, ts) VALUES (?,?,?,?,?,?,?,?)",
                      (pid, kind, name, json.dumps(config), json.dumps(result), json.dumps(summary), context, time.time()))

    def run(self, pid, rid) -> dict | None:
        r = self.one("SELECT * FROM runs WHERE id=? AND player_id=?", (rid, pid))
        if not r:
            return None
        d = dict(r)
        for k in ("config", "result", "summary"):
            d[k] = json.loads(d[k])
        return d

    def runs(self, pid, kind=None, limit=200) -> list[dict]:
        sql = "SELECT id, kind, name, config, summary, notes, context, ts FROM runs WHERE player_id=?"
        args: tuple = (pid,)
        if kind:
            sql += " AND kind=?"
            args += (kind,)
        rows = self.q(sql + " ORDER BY id DESC LIMIT ?", args + (limit,))
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d["config"]); d["summary"] = json.loads(d["summary"])
            out.append(d)
        return out

    def update_run(self, pid, rid, **fields):
        bad = set(fields) - {"name", "notes", "summary"}
        if bad:
            raise ValueError(f"Cannot update run fields {bad}")
        if "summary" in fields:
            fields["summary"] = json.dumps(fields["summary"])
        sets = ", ".join(f"{k}=?" for k in fields)
        self.x(f"UPDATE runs SET {sets} WHERE id=? AND player_id=?", (*fields.values(), rid, pid))

    # -- progress
    def progress(self, pid, kind) -> dict[str, dict]:
        return {r["item_id"]: dict(json.loads(r["state"]), completed_at=r["completed_at"])
                for r in self.q("SELECT item_id, state, completed_at FROM progress WHERE player_id=? AND kind=?", (pid, kind))}

    def set_progress(self, pid, kind, item_id, state: dict, completed: bool = False):
        st = {k: v for k, v in state.items() if k != "completed_at"}
        prev = self.one("SELECT completed_at FROM progress WHERE player_id=? AND kind=? AND item_id=?", (pid, kind, item_id))
        done_at = (prev["completed_at"] if prev and prev["completed_at"] else (time.time() if completed else None))
        self.x("INSERT INTO progress(player_id, kind, item_id, state, completed_at) VALUES (?,?,?,?,?) "
               "ON CONFLICT(player_id, kind, item_id) DO UPDATE SET state=excluded.state, completed_at=excluded.completed_at",
               (pid, kind, item_id, json.dumps(st), done_at))

    # -- achievements
    def achievements(self, pid) -> dict[str, float]:
        return {r["ach_id"]: r["ts"] for r in self.q("SELECT ach_id, ts FROM achievements WHERE player_id=?", (pid,))}

    def grant(self, pid, ach_id) -> bool:
        if self.one("SELECT 1 FROM achievements WHERE player_id=? AND ach_id=?", (pid, ach_id)):
            return False
        self.x("INSERT INTO achievements(player_id, ach_id, ts) VALUES (?,?,?)", (pid, ach_id, time.time()))
        return True

    def events(self, pid, limit=30):
        return [dict(r) for r in self.q("SELECT xp, reason, ts FROM events WHERE player_id=? ORDER BY id DESC LIMIT ?", (pid, limit))]

    def add_reflection(self, pid, cid, text):
        self.x("INSERT INTO reflections(player_id, concept_id, text, ts) VALUES (?,?,?,?)", (pid, cid, text[:2000], time.time()))

    def reflections(self, pid):
        return [dict(r) for r in self.q("SELECT concept_id, text, ts FROM reflections WHERE player_id=? ORDER BY id DESC", (pid,))]
