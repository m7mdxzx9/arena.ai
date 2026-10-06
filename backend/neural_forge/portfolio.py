"""Portfolio case studies grounded in saved experiment runs."""
from __future__ import annotations

import html
import json
import time
import uuid
from typing import Any

from .db import DB

FIELDS = ("title", "problem", "dataset", "method", "metrics", "interpretation", "limitations", "next_steps")


def _public(row: Any) -> dict[str, Any]:
    content = json.loads(row["content"])
    return {"id": row["id"], "created_at": row["created_at"], "updated_at": row["updated_at"], **content}


def list_projects(db: DB, player_id: int) -> list[dict[str, Any]]:
    return [_public(row) for row in db.q("SELECT * FROM portfolio_projects WHERE player_id=? ORDER BY updated_at DESC", (player_id,))]


def get_project(db: DB, player_id: int, project_id: str) -> dict[str, Any]:
    row = db.one("SELECT * FROM portfolio_projects WHERE id=? AND player_id=?", (project_id, player_id))
    if not row:
        raise KeyError(project_id)
    return _public(row)


def create_from_run(db: DB, player_id: int, run_id: int, content: dict[str, Any]) -> dict[str, Any]:
    run = db.run(player_id, run_id)
    if not run:
        raise ValueError("Saved experiment run was not found.")
    project_id = uuid.uuid4().hex
    summary = run.get("summary") or {}
    config = run.get("config") or {}
    project = {
        "title": str(content.get("title") or run.get("name") or f"Experiment {run_id}")[:160],
        "problem": str(content.get("problem", ""))[:10_000],
        "dataset": str(content.get("dataset") or config.get("dataset") or config.get("dataset_id") or "")[:2_000],
        "method": str(content.get("method") or config.get("model") or run.get("kind") or "")[:10_000],
        "metrics": content.get("metrics") if content.get("metrics") not in (None, "") else summary,
        "interpretation": str(content.get("interpretation", ""))[:10_000],
        "limitations": str(content.get("limitations", ""))[:10_000],
        "next_steps": str(content.get("next_steps", ""))[:10_000],
        "source_run_id": run_id,
        "source_run_kind": run.get("kind"),
        "source_snapshot": {"config": config, "summary": summary},
    }
    now = time.time()
    db.x("INSERT INTO portfolio_projects(id, player_id, title, content, created_at, updated_at) VALUES (?,?,?,?,?,?)", (project_id, player_id, project["title"], json.dumps(project, ensure_ascii=False), now, now))
    return get_project(db, player_id, project_id)


def update_project(db: DB, player_id: int, project_id: str, content: dict[str, Any]) -> dict[str, Any]:
    current = get_project(db, player_id, project_id)
    for field in FIELDS:
        if field in content:
            current[field] = content[field] if field == "metrics" else str(content[field])[:10_000]
    if not str(current.get("title", "")).strip():
        raise ValueError("Project title is required.")
    db.x("UPDATE portfolio_projects SET title=?, content=?, updated_at=? WHERE id=? AND player_id=?", (str(current["title"])[:160], json.dumps({key: value for key, value in current.items() if key not in {"id", "created_at", "updated_at"}}, ensure_ascii=False), time.time(), project_id, player_id))
    return get_project(db, player_id, project_id)


def to_markdown(project: dict[str, Any]) -> str:
    metrics = project.get("metrics", "")
    metrics_text = "```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```" if not isinstance(metrics, str) else metrics
    return f"""# {project.get('title', 'AI Project')}

> Source experiment: #{project.get('source_run_id', '—')} · `{project.get('source_run_kind', 'experiment')}`

## Problem

{project.get('problem') or 'Not documented.'}

## Dataset

{project.get('dataset') or 'Not documented.'}

## Method

{project.get('method') or 'Not documented.'}

## Metrics

{metrics_text}

## Interpretation

{project.get('interpretation') or 'Not documented.'}

## Limitations

{project.get('limitations') or 'Not documented.'}

## Next steps

{project.get('next_steps') or 'Not documented.'}
"""


def export_project(project: dict[str, Any], format: str) -> tuple[str, str]:
    if format == "json":
        return json.dumps(project, ensure_ascii=False, indent=2), "application/json"
    markdown = to_markdown(project)
    if format == "markdown":
        return markdown, "text/markdown"
    if format == "html":
        sections = []
        for heading, field in (("Problem", "problem"), ("Dataset", "dataset"), ("Method", "method"), ("Metrics", "metrics"), ("Interpretation", "interpretation"), ("Limitations", "limitations"), ("Next steps", "next_steps")):
            value = project.get(field, "")
            text = json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
            sections.append(f"<section><h2>{heading}</h2><pre>{html.escape(text)}</pre></section>")
        document = "<!doctype html><html><head><meta charset='utf-8'><title>" + html.escape(str(project.get("title", "AI Project"))) + "</title><style>body{max-width:800px;margin:auto;padding:2rem;font:16px system-ui;line-height:1.6}pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style></head><body><h1>" + html.escape(str(project.get("title", "AI Project"))) + "</h1>" + "".join(sections) + "</body></html>"
        return document, "text/html"
    raise ValueError("Export format must be markdown, html, or json.")
