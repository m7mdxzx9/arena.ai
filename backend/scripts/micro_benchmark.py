#!/usr/bin/env python3
"""Small repeatable benchmark for the deterministic ranking utility."""
from __future__ import annotations

import json
import math
import platform
import statistics
import time

from neural_forge.ranking import top_five_scores


def main() -> None:
    records = [(f"label-{index % 997:04d}", math.sin(index) * 100 + (index % 17)) for index in range(25_000)]
    durations = []
    result = []
    for _ in range(25):
        started = time.perf_counter()
        result = top_five_scores(records)
        durations.append((time.perf_counter() - started) * 1_000)
    print(json.dumps({
        "python": platform.python_version(),
        "input_records": len(records),
        "iterations": len(durations),
        "median_ms": round(statistics.median(durations), 3),
        "p95_ms": round(sorted(durations)[int(len(durations) * 0.95) - 1], 3),
        "result": result,
    }, indent=2))


if __name__ == "__main__":
    main()
