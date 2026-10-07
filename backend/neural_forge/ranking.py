"""Small ranking utilities shared by retrieval and evaluation views."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import SupportsFloat


def top_five_scores(items: Iterable[tuple[str, SupportsFloat]]) -> list[tuple[str, float]]:
    """Return the five highest finite scores, keeping each label's maximum.

    Empty labels and values that cannot be converted to a finite float are ignored.
    Ties are stable by alphabetic (case-insensitive, then exact) label order.
    """
    best: dict[str, float] = {}
    for raw_label, raw_score in items:
        label = str(raw_label)
        if not label:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(score):
            continue
        if label not in best or score > best[label]:
            best[label] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[:5]
