"""Tiny DSL used by the curriculum content modules.

A *concept* is the atomic unit of mastery. Every concept has:
  * a short explanation (teach card) and an analogy,
  * an optional interactive visual (frontend widget key),
  * an optional worked example (demonstrate card),
  * questions at difficulty 1 (easy), 2 (medium) and 3 (hard),
  * optional parametric question generators (unlimited, freshly computed practice),
  * optional link to a predict-before-running experiment,
  * a reflection prompt.
"""
from __future__ import annotations

from typing import Any


def Q(d: int, q: str, opts: list[str], a: int, why: str, hint: str = "") -> dict[str, Any]:
    """Multiple-choice question. `a` is the index of the correct option."""
    assert 0 <= a < len(opts), q
    assert d in (1, 2, 3)
    return {"type": "mcq", "difficulty": d, "prompt": q, "options": opts, "answer": a,
            "explanation": why, "hint": hint}


def N(d: int, q: str, answer: float, tol: float, why: str, hint: str = "") -> dict[str, Any]:
    """Numeric-answer question, graded with absolute tolerance `tol`."""
    return {"type": "numeric", "difficulty": d, "prompt": q, "answer": answer, "tolerance": tol,
            "explanation": why, "hint": hint}


def C(cid: str, name: str, branch: str, area: str, prereqs: list[str], *, explain: str,
      analogy: str = "", visual: str | None = None, example: str = "", questions: list[dict] = (),
      generators: list[str] = (), predict: str | None = None, reflect: str = "",
      code: str = "") -> dict[str, Any]:
    return {
        "id": cid, "name": name, "branch": branch, "area": area, "prereqs": list(prereqs),
        "explain": explain.strip(), "analogy": analogy.strip(), "visual": visual,
        "example": example.strip(), "questions": list(questions), "generators": list(generators),
        "predict": predict, "reflect": reflect.strip(), "code": code.strip(),
    }
