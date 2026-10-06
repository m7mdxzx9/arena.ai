"""Central deterministic evaluation engine for prompts, retrieval, tools and labels."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import defaultdict
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .db import DB

EVALUATOR_TYPES = {
    "exact_match", "contains", "regex", "numeric_tolerance", "json_schema",
    "classification_label", "citation_presence", "tool_selection", "retrieval_contains",
}


def _safe_regex(pattern: str) -> re.Pattern[str]:
    if len(pattern) > 500:
        raise ValueError("Regex patterns are limited to 500 characters.")
    # Reject common catastrophic nested-quantifier forms. This is intentionally
    # conservative because evaluation data is untrusted user input.
    if re.search(r"\([^)]*[+*][^)]*\)[+*{]", pattern) or re.search(r"(\.\*){2,}|(\.\+){2,}", pattern):
        raise ValueError("Regex contains a potentially unsafe nested quantifier.")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc


def validate_case(case: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError(f"Evaluation case {index + 1} must be an object.")
    evaluator = case.get("evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("type") not in EVALUATOR_TYPES:
        raise ValueError(f"Evaluation case {index + 1} has an unknown evaluator type.")
    if evaluator["type"] == "regex":
        _safe_regex(str(evaluator.get("pattern", case.get("reference_answer", ""))))
    if evaluator["type"] == "json_schema":
        try:
            Draft202012Validator.check_schema(evaluator.get("schema", {}))
        except SchemaError as exc:
            raise ValueError(f"Evaluation case {index + 1} contains an invalid JSON schema.") from exc
    clean = {
        "id": str(case.get("id") or f"case-{index + 1}")[:100],
        "input": case.get("input", ""),
        "expected_behavior": str(case.get("expected_behavior", ""))[:2_000],
        "reference_answer": case.get("reference_answer"),
        "category": str(case.get("category", "general"))[:100],
        "difficulty": str(case.get("difficulty", "standard"))[:50],
        "tags": [str(tag)[:50] for tag in case.get("tags", [])[:20]],
        "metadata": case.get("metadata", {}) if isinstance(case.get("metadata", {}), dict) else {},
        "evaluator": evaluator,
    }
    if len(json.dumps(clean, ensure_ascii=False)) > 50_000:
        raise ValueError(f"Evaluation case {index + 1} is too large.")
    return clean


def create_dataset(db: DB, player_id: int, name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not 1 <= len(cases) <= 500:
        raise ValueError("An evaluation dataset must contain 1–500 cases.")
    validated = [validate_case(case, index) for index, case in enumerate(cases)]
    ids = [case["id"] for case in validated]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation case IDs must be unique.")
    dataset_id = uuid.uuid4().hex
    now = time.time()
    db.x(
        "INSERT INTO evaluation_datasets(id, player_id, name, cases_json, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (dataset_id, player_id, name.strip()[:100] or "Evaluation Dataset", json.dumps(validated, ensure_ascii=False), now, now),
    )
    return get_dataset(db, player_id, dataset_id)


def list_datasets(db: DB, player_id: int) -> list[dict[str, Any]]:
    rows = db.q("SELECT id, name, cases_json, created_at, updated_at FROM evaluation_datasets WHERE player_id=? ORDER BY updated_at DESC", (player_id,))
    return [{"id": row["id"], "name": row["name"], "cases": len(json.loads(row["cases_json"])), "created_at": row["created_at"], "updated_at": row["updated_at"]} for row in rows]


def get_dataset(db: DB, player_id: int, dataset_id: str) -> dict[str, Any]:
    row = db.one("SELECT * FROM evaluation_datasets WHERE id=? AND player_id=?", (dataset_id, player_id))
    if not row:
        raise KeyError(dataset_id)
    return {"id": row["id"], "name": row["name"], "cases": json.loads(row["cases_json"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def evaluate_case(case: dict[str, Any], actual: Any) -> dict[str, Any]:
    spec = case["evaluator"]
    kind = spec["type"]
    reference = spec.get("expected", case.get("reference_answer"))
    passed = False
    detail = ""
    try:
        if kind in {"exact_match", "classification_label"}:
            left, right = _text(actual), _text(reference)
            if spec.get("strip", True):
                left, right = left.strip(), right.strip()
            if spec.get("case_sensitive", False) is False:
                left, right = left.casefold(), right.casefold()
            passed, detail = left == right, f"expected {right!r}; got {left!r}"
        elif kind == "contains":
            haystack = _text(actual)
            needles = reference if isinstance(reference, list) else [reference]
            if not spec.get("case_sensitive", False):
                haystack, needles = haystack.casefold(), [str(item).casefold() for item in needles]
            passed = all(str(item) in haystack for item in needles)
            detail = f"required substrings: {needles}"
        elif kind == "regex":
            pattern = _safe_regex(str(spec.get("pattern", reference or "")))
            passed = bool(pattern.search(_text(actual)[:20_000]))
            detail = f"pattern: {pattern.pattern}"
        elif kind == "numeric_tolerance":
            expected = float(reference)
            value = float(actual)
            absolute = float(spec.get("absolute", spec.get("tolerance", 1e-6)))
            relative = float(spec.get("relative", 0.0))
            passed = math.isfinite(value) and math.isclose(value, expected, rel_tol=max(relative, 0), abs_tol=max(absolute, 0))
            detail = f"expected {expected} ± abs {absolute}, rel {relative}; got {value}"
        elif kind == "json_schema":
            value = json.loads(actual) if isinstance(actual, str) else actual
            validator = Draft202012Validator(spec.get("schema", {}))
            errors = list(validator.iter_errors(value))
            passed = not errors
            detail = "valid JSON schema" if passed else "; ".join(error.message for error in errors[:3])
        elif kind == "citation_presence":
            text = _text(actual)
            minimum = int(spec.get("minimum", 1))
            citations = re.findall(r"\[[^\[\]\n]{1,120}\]", text)
            passed, detail = len(citations) >= minimum, f"found {len(citations)} citation(s), need {minimum}"
        elif kind == "tool_selection":
            value = json.loads(actual) if isinstance(actual, str) else actual
            selected = value.get("tool") if isinstance(value, dict) else None
            passed, detail = selected == reference, f"expected tool {reference!r}; got {selected!r}"
        elif kind == "retrieval_contains":
            values = actual if isinstance(actual, list) else actual.get("retrieved", []) if isinstance(actual, dict) else []
            ids = {item.get("id") if isinstance(item, dict) else item for item in values}
            expected_ids = set(reference if isinstance(reference, list) else [reference])
            passed, detail = expected_ids <= ids, f"missing IDs: {sorted(expected_ids - ids)}"
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        passed, detail = False, f"Evaluator could not parse the output: {str(exc)[:300]}"
    return {"case_id": case["id"], "evaluator": kind, "kind": "deterministic", "passed": bool(passed), "score": 1.0 if passed else 0.0, "detail": detail, "actual": actual}


def run_evaluation(db: DB, player_id: int, dataset_id: str, outputs: dict[str, Any], name: str | None = None) -> dict[str, Any]:
    dataset = get_dataset(db, player_id, dataset_id)
    results = [evaluate_case(case, outputs.get(case["id"])) for case in dataset["cases"]]
    by_category: dict[str, list[float]] = defaultdict(list)
    case_lookup = {case["id"]: case for case in dataset["cases"]}
    for result in results:
        by_category[case_lookup[result["case_id"]]["category"]].append(result["score"])
    passed = sum(result["passed"] for result in results)
    summary = {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "score": round(passed / len(results), 4),
        "by_category": {category: round(sum(scores) / len(scores), 4) for category, scores in by_category.items()},
        "evaluator_kind": "deterministic",
    }
    payload = {"dataset_id": dataset_id, "dataset_name": dataset["name"], "summary": summary, "results": results}
    run_id = db.save_run(player_id, "evaluation", {"dataset_id": dataset_id}, payload, summary, name=name or dataset["name"], context="evaluation_lab")
    return {"run_id": run_id, **payload}
