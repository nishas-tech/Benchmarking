"""Pandas aggregation for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from local_slm_benchmark.benchmark.results import load_results
from local_slm_benchmark.evaluation.deepeval_runner import score_result
from local_slm_benchmark.evaluation.diff import compute_variance_scores
from local_slm_benchmark.models.schemas import BenchmarkResult


@dataclass(frozen=True)
class AnalysisOutput:
    results: list[BenchmarkResult]
    dataframe: pd.DataFrame
    summary: pd.DataFrame


def analyze_results(results_path: str | Path) -> AnalysisOutput:
    results = load_results(results_path)
    variance_scores = compute_variance_scores(results)
    enriched: list[BenchmarkResult] = []
    for result in results:
        enriched.append(
            result.model_copy(
                update={
                    "quality_score": score_result(result),
                    "variance_score": variance_scores.get(result.run_id),
                }
            )
        )

    dataframe = results_to_dataframe(enriched)
    summary = summarize_dataframe(dataframe)
    return AnalysisOutput(results=enriched, dataframe=dataframe, summary=summary)


def results_to_dataframe(results: list[BenchmarkResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "run_id": result.run_id,
                "model": result.model,
                "temperature": result.temperature,
                "prompt_id": result.prompt_id,
                "prompt_category": result.prompt_category,
                "valid_json": result.valid_json,
                "retry_count": result.retry_count,
                "time_to_first_token_ms": result.time_to_first_token_ms,
                "total_latency_ms": result.total_latency_ms,
                "tokens_per_second": result.tokens_per_second,
                "memory_mb": result.memory_mb,
                "quality_score": result.quality_score,
                "variance_score": result.variance_score,
            }
        )
    return pd.DataFrame(rows)


def summarize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame()

    grouped = dataframe.groupby(["model", "temperature"], dropna=False)
    summary = grouped.agg(
        average_latency_ms=("total_latency_ms", "mean"),
        p95_latency_ms=("total_latency_ms", lambda series: series.quantile(0.95)),
        average_tokens_per_second=("tokens_per_second", "mean"),
        average_memory_mb=("memory_mb", "mean"),
        json_validation_success_rate=("valid_json", "mean"),
        retry_rate=("retry_count", lambda series: (series > 0).mean()),
        quality_score=("quality_score", "mean"),
        output_variance_score=("variance_score", "mean"),
        run_count=("run_id", "count"),
    )
    return summary.reset_index()

