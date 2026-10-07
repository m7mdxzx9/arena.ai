"""Persistent, reviewable learning mistakes connected to real player actions."""
from __future__ import annotations

import time
from typing import Any

from .db import DB

DAY = 86_400
REVIEW_INTERVALS = [DAY, 3 * DAY, 7 * DAY, 16 * DAY, 35 * DAY]

TEMPLATES: dict[str, dict[str, str]] = {
    "overfit": {
        "concept": "overfitting",
        "category": "model_evaluation",
        "mistake_type": "overfit_model",
        "correct_principle": "Prefer validation performance and generalisation over training performance.",
        "explanation": "A large train–validation gap means the model learned training-specific patterns that do not transfer.",
        "example": "Reduce tree depth, add regularisation, remove noisy features, gather data, or use early stopping.",
    },
    "too_good": {
        "concept": "data_leakage",
        "category": "data_preparation",
        "mistake_type": "possible_data_leakage",
        "correct_principle": "Every feature must be available at prediction time and independent of the future target.",
        "explanation": "A near-perfect realistic score can indicate a feature derived from, or measured after, the outcome.",
        "example": "For default prediction at application time, collection calls made after default are leakage.",
    },
    "imbalance": {
        "concept": "class_imbalance",
        "category": "metrics",
        "mistake_type": "accuracy_on_imbalanced_data",
        "correct_principle": "Evaluate the minority class with recall, precision, F1, PR curves, and task-specific cost.",
        "explanation": "Accuracy can look high when a model simply predicts the majority class.",
        "example": "At 2% fraud, predicting “not fraud” for every row is 98% accurate but catches no fraud.",
    },
    "unscaled": {
        "concept": "feature_scaling",
        "category": "preprocessing",
        "mistake_type": "distance_model_without_scaling",
        "correct_principle": "Fit scaling on training data and apply the learned transformation to validation/test data.",
        "explanation": "Distance-based models are dominated by features with the largest numeric range.",
        "example": "Standardise income and visit count before k-means or k-nearest neighbours.",
    },
    "worse_than_mean": {
        "concept": "r2",
        "category": "metrics",
        "mistake_type": "ignored_baseline",
        "correct_principle": "Always compare a trained model with a simple baseline on the same held-out data.",
        "explanation": "Negative test R² means predicting the training mean would have performed better.",
        "example": "Investigate preprocessing and features before increasing model complexity.",
    },
}


def _row(row: Any) -> dict[str, Any]:
    result = dict(row)
    result["resolved"] = bool(result["resolved"])
    result["due"] = not result["resolved"] and result["due_at"] <= time.time()
    return result


def create(
    db: DB,
    player_id: int,
    *,
    concept: str,
    category: str,
    mistake_type: str,
    player_action: str,
    correct_principle: str,
    explanation: str,
    example: str = "",
    mission_id: str | None = None,
    run_id: int | None = None,
) -> dict[str, Any]:
    fields = [concept, category, mistake_type, player_action, correct_principle, explanation, example]
    if any(len(str(value)) > 2_000 for value in fields):
        raise ValueError("Mistake fields must be at most 2,000 characters.")
    if run_id is not None:
        existing = db.one(
            "SELECT * FROM mistakes WHERE player_id=? AND run_id=? AND mistake_type=?",
            (player_id, run_id, mistake_type),
        )
        if existing:
            return _row(existing)
    now = time.time()
    mistake_id = db.x(
        "INSERT INTO mistakes(player_id, concept, category, mission_id, run_id, mistake_type, player_action, "
        "correct_principle, explanation, example, created_at, review_count, resolved, due_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            player_id,
            concept[:120],
            category[:120],
            mission_id,
            run_id,
            mistake_type[:120],
            player_action[:2_000],
            correct_principle[:2_000],
            explanation[:2_000],
            example[:2_000],
            now,
            0,
            0,
            now,
        ),
    )
    return _row(db.one("SELECT * FROM mistakes WHERE id=?", (mistake_id,)))


def record_ml_diagnostics(db: DB, player_id: int, run_id: int, config: dict, result: dict) -> list[dict[str, Any]]:
    """Create records only for diagnostics caused by the player's actual run."""
    output = []
    model = str(config.get("model", "model"))
    features = ", ".join(config.get("features", [])[:12])
    for diagnosis in result.get("diagnosis", []):
        code = diagnosis.get("code") if isinstance(diagnosis, dict) else None
        template = TEMPLATES.get(str(code))
        if not template:
            continue
        # Imbalance is educationally meaningful as a mistake only when the run did
        # not opt into balanced class weighting. The diagnosis itself still appears.
        if code == "imbalance" and config.get("params", {}).get("class_weight") == "balanced":
            continue
        action = f"Trained {model} on features [{features}]"
        output.append(
            create(
                db,
                player_id,
                run_id=run_id,
                player_action=action,
                **template,
            )
        )
    return output


def list_records(db: DB, player_id: int, status: str = "all", topic: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM mistakes WHERE player_id=?"
    args: list[Any] = [player_id]
    if status == "unresolved":
        sql += " AND resolved=0"
    elif status == "mastered":
        sql += " AND resolved=1"
    elif status == "due":
        sql += " AND resolved=0 AND due_at<=?"
        args.append(time.time())
    if topic:
        sql += " AND (concept=? OR category=?)"
        args.extend([topic, topic])
    sql += " ORDER BY created_at DESC LIMIT 500"
    return [_row(row) for row in db.q(sql, tuple(args))]


def review(db: DB, player_id: int, mistake_id: int, remembered: bool) -> dict[str, Any]:
    row = db.one("SELECT * FROM mistakes WHERE id=? AND player_id=?", (mistake_id, player_id))
    if not row:
        raise KeyError(mistake_id)
    count = int(row["review_count"]) + 1 if remembered else max(0, int(row["review_count"]) - 1)
    resolved = remembered and count >= 3
    interval = REVIEW_INTERVALS[min(count, len(REVIEW_INTERVALS) - 1)] if remembered else DAY
    db.x(
        "UPDATE mistakes SET review_count=?, resolved=?, due_at=? WHERE id=? AND player_id=?",
        (count, int(resolved), time.time() + interval, mistake_id, player_id),
    )
    return _row(db.one("SELECT * FROM mistakes WHERE id=?", (mistake_id,)))
