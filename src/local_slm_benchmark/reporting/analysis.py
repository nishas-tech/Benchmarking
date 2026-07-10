"""Pandas aggregation for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from local_slm_benchmark.benchmark.metadata import BenchmarkRunMetadata, load_run_metadata
from local_slm_benchmark.benchmark.results import load_results
from local_slm_benchmark.config import load_benchmark_config
from local_slm_benchmark.evaluation.deepeval_runner import evaluate_quality
from local_slm_benchmark.evaluation.diff import compute_temperature_comparisons, compute_variance_scores, summarize_temperature_comparisons
from local_slm_benchmark.models.schemas import BenchmarkResult
from local_slm_benchmark.prompts import load_prompts


@dataclass(frozen=True)
class ReliabilitySummary:
    json_parse_success_rate: float
    schema_validation_success_rate: float
    retry_rate: float
    retry_success_rate: float
    final_failure_rate: float
    schema_error_categories: dict[str, int]


@dataclass(frozen=True)
class AnalysisOutput:
    results: list[BenchmarkResult]
    dataframe: pd.DataFrame
    summary: pd.DataFrame
    reliability: ReliabilitySummary
    temperature_comparison: pd.DataFrame
    quality_dimensions: pd.DataFrame
    metadata: BenchmarkRunMetadata | None = None
    prompt_summary: dict[str, Any] = field(default_factory=dict)


def analyze_results(
    results_path: str | Path,
    *,
    quality_scorer: str | None = None,
) -> AnalysisOutput:
    config = load_benchmark_config()
    prompts = load_prompts(config.prompts_path)
    prompts_by_id = {prompt.id: prompt.prompt for prompt in prompts}
    results = load_results(results_path)
    variance_scores = compute_variance_scores(results)
    enriched: list[BenchmarkResult] = []
    for result in results:
        quality = evaluate_quality(
            result,
            prompt_text=prompts_by_id.get(result.prompt_id),
            scorer=quality_scorer,
        )
        extra = dict(result.extra)
        extra["quality_scorer"] = quality.scorer
        if quality.reason:
            extra["quality_reason"] = quality.reason
        if quality.dimension_reasons:
            extra["quality_dimension_reasons"] = quality.dimension_reasons
        enriched.append(
            result.model_copy(
                update={
                    "quality_score": quality.score,
                    "quality_dimensions": quality.dimensions,
                    "variance_score": variance_scores.get(result.run_id),
                    "extra": extra,
                }
            )
        )

    dataframe = results_to_dataframe(enriched)
    summary = summarize_dataframe(dataframe)
    reliability = summarize_reliability(dataframe)
    temperature_comparison = pd.DataFrame(summarize_temperature_comparisons(compute_temperature_comparisons(enriched)))
    quality_dimensions = summarize_quality_dimensions(enriched)
    metadata = load_run_metadata(results_path)
    prompt_summary = summarize_prompts(prompts)
    return AnalysisOutput(
        results=enriched,
        dataframe=dataframe,
        summary=summary,
        reliability=reliability,
        temperature_comparison=temperature_comparison,
        quality_dimensions=quality_dimensions,
        metadata=metadata,
        prompt_summary=prompt_summary,
    )


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
                "valid_json_parse": result.valid_json_parse,
                "valid_schema": result.valid_schema,
                "schema_error_category": result.schema_error_category,
                "retry_count": result.retry_count,
                "retry_succeeded": result.retry_succeeded,
                "final_failure": result.final_failure,
                "time_to_first_token_ms": result.time_to_first_token_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "retry_latency_ms": result.retry_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "tokens_per_second": result.tokens_per_second,
                "output_tokens": result.output_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "memory_mb": result.memory_mb,
                "model_quantization": result.model_quantization,
                "gpu_memory_used_mb": result.gpu_memory_used_mb,
                "gpu_utilization_percent": result.gpu_utilization_percent,
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
        average_generation_latency_ms=("generation_latency_ms", "mean"),
        average_retry_latency_ms=("retry_latency_ms", "mean"),
        p95_latency_ms=("total_latency_ms", lambda series: series.quantile(0.95)),
        average_time_to_first_token_ms=("time_to_first_token_ms", "mean"),
        average_tokens_per_second=("tokens_per_second", "mean"),
        average_memory_mb=("memory_mb", "mean"),
        average_gpu_memory_used_mb=("gpu_memory_used_mb", "mean"),
        json_validation_success_rate=("valid_json", "mean"),
        json_parse_success_rate=("valid_json_parse", "mean"),
        schema_validation_success_rate=("valid_schema", "mean"),
        retry_rate=("retry_count", lambda series: (series > 0).mean()),
        retry_success_rate=("retry_succeeded", "mean"),
        final_failure_rate=("final_failure", lambda series: series.notna().mean()),
        quality_score=("quality_score", "mean"),
        output_variance_score=("variance_score", "mean"),
        run_count=("run_id", "count"),
    )
    return summary.reset_index()


def summarize_reliability(dataframe: pd.DataFrame) -> ReliabilitySummary:
    if dataframe.empty:
        return ReliabilitySummary(0, 0, 0, 0, 0, {})

    categories = (
        dataframe["schema_error_category"]
        .fillna("none")
        .value_counts()
        .to_dict()
    )
    return ReliabilitySummary(
        json_parse_success_rate=float(dataframe["valid_json_parse"].mean()),
        schema_validation_success_rate=float(dataframe["valid_schema"].mean()),
        retry_rate=float((dataframe["retry_count"] > 0).mean()),
        retry_success_rate=float(dataframe.loc[dataframe["retry_count"] > 0, "retry_succeeded"].mean())
        if (dataframe["retry_count"] > 0).any()
        else 0.0,
        final_failure_rate=float(dataframe["final_failure"].notna().mean()),
        schema_error_categories={str(key): int(value) for key, value in categories.items()},
    )


def summarize_quality_dimensions(results: list[BenchmarkResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if not result.quality_dimensions:
            continue
        row = {
            "model": result.model,
            "temperature": result.temperature,
            "prompt_id": result.prompt_id,
        }
        row.update(result.quality_dimensions)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    dataframe = pd.DataFrame(rows)
    dimension_columns = [column for column in dataframe.columns if column not in {"model", "temperature", "prompt_id"}]
    grouped = dataframe.groupby(["model", "temperature"], dropna=False)
    return grouped[dimension_columns].mean().reset_index()


def summarize_prompts(prompts) -> dict[str, Any]:
    categories: dict[str, int] = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1
    return {
        "prompt_count": len(prompts),
        "categories": categories,
    }
