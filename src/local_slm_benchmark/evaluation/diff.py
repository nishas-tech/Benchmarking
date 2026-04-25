"""DeepDiff-based output variance analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from deepdiff import DeepDiff

from local_slm_benchmark.models.schemas import BenchmarkResult


def diff_score(diff: DeepDiff) -> float:
    if not diff:
        return 0.0
    return float(sum(len(value) if hasattr(value, "__len__") else 1 for value in diff.values()))


def compute_variance_scores(results: list[BenchmarkResult]) -> dict[str, float]:
    grouped: dict[tuple[str, float, str], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[(result.model, result.temperature, result.prompt_id)].append(result)

    scores: dict[str, float] = {}
    for rows in grouped.values():
        baseline = _comparable_payload(rows[0])
        for row in rows:
            diff = DeepDiff(baseline, _comparable_payload(row), ignore_order=True)
            scores[row.run_id] = diff_score(diff)
    return scores


def _comparable_payload(result: BenchmarkResult) -> dict[str, Any] | str:
    return result.parsed_response or result.raw_response

