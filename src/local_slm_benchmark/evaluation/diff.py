"""DeepDiff-based output variance analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from deepdiff import DeepDiff

from local_slm_benchmark.models.schemas import BenchmarkResult


@dataclass(frozen=True)
class TemperatureComparison:
    model: str
    prompt_id: str
    low_temperature: float
    high_temperature: float
    low_temperature_variance: float
    high_temperature_variance: float
    cross_temperature_distance: float


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


def compute_temperature_comparisons(results: list[BenchmarkResult]) -> list[TemperatureComparison]:
    if not results:
        return []

    temperatures = sorted({result.temperature for result in results})
    if len(temperatures) < 2:
        return []

    low_temperature = temperatures[0]
    high_temperature = temperatures[-1]
    grouped: dict[tuple[str, str], dict[float, list[BenchmarkResult]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        grouped[(result.model, result.prompt_id)][result.temperature].append(result)

    variance_scores = compute_variance_scores(results)
    comparisons: list[TemperatureComparison] = []
    for (model, prompt_id), by_temperature in grouped.items():
        low_rows = by_temperature.get(low_temperature, [])
        high_rows = by_temperature.get(high_temperature, [])
        if not low_rows or not high_rows:
            continue
        low_variance = _average_variance(low_rows, variance_scores)
        high_variance = _average_variance(high_rows, variance_scores)
        cross_diff = DeepDiff(
            _comparable_payload(low_rows[0]),
            _comparable_payload(high_rows[0]),
            ignore_order=True,
        )
        comparisons.append(
            TemperatureComparison(
                model=model,
                prompt_id=prompt_id,
                low_temperature=low_temperature,
                high_temperature=high_temperature,
                low_temperature_variance=low_variance,
                high_temperature_variance=high_variance,
                cross_temperature_distance=diff_score(cross_diff),
            )
        )
    return comparisons


def summarize_temperature_comparisons(comparisons: list[TemperatureComparison]) -> list[dict[str, float | str]]:
    grouped: dict[tuple[str, float, float], list[TemperatureComparison]] = defaultdict(list)
    for comparison in comparisons:
        grouped[(comparison.model, comparison.low_temperature, comparison.high_temperature)].append(comparison)

    rows: list[dict[str, float | str]] = []
    for (model, low_temperature, high_temperature), items in grouped.items():
        rows.append(
            {
                "model": model,
                "low_temperature": low_temperature,
                "high_temperature": high_temperature,
                "average_low_temp_variance": _mean(item.low_temperature_variance for item in items),
                "average_high_temp_variance": _mean(item.high_temperature_variance for item in items),
                "average_cross_temperature_distance": _mean(item.cross_temperature_distance for item in items),
                "prompt_count": float(len(items)),
            }
        )
    return rows


def _average_variance(rows: list[BenchmarkResult], variance_scores: dict[str, float]) -> float:
    values = [variance_scores.get(row.run_id, 0.0) for row in rows]
    return _mean(values)


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _comparable_payload(result: BenchmarkResult) -> dict[str, Any] | str:
    return result.parsed_response or result.raw_response
