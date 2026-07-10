"""Markdown report and chart generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from local_slm_benchmark.benchmark.results import latest_results_file
from local_slm_benchmark.config import load_benchmark_config, project_path
from local_slm_benchmark.observability.analysis_exporter import export_analysis_metrics
from local_slm_benchmark.observability.persistent_metrics import persist_analysis_summary
from local_slm_benchmark.reporting.analysis import AnalysisOutput, analyze_results
from local_slm_benchmark.reporting.recommendation import ModelRecommendation, recommend_models


def generate_report(
    results_path: str | Path | None = None,
    *,
    quality_scorer: str | None = None,
) -> Path:
    config = load_benchmark_config()
    selected_results = Path(results_path) if results_path else latest_results_file(config.results_dir)
    if selected_results is None:
        raise FileNotFoundError("No benchmark result files found.")

    analysis = analyze_results(selected_results, quality_scorer=quality_scorer)
    export_analysis_metrics(analysis.summary)
    persist_analysis_summary(analysis.summary)

    reports_dir = project_path(config.reports_dir)
    charts_dir = reports_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    recommendations = recommend_models(analysis.summary)
    chart_paths = _generate_charts(analysis, charts_dir)
    report_path = reports_dir / f"model-comparison-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.md"
    report_path.write_text(
        _render_markdown(
            selected_results,
            analysis,
            chart_paths,
            recommendations,
            quality_scorer or config.quality_scorer,
        ),
        encoding="utf-8",
    )
    return report_path


def _generate_charts(analysis: AnalysisOutput, charts_dir: Path) -> dict[str, Path]:
    if analysis.summary.empty:
        return {}

    chart_paths = {
        "average_latency": charts_dir / "average-latency-by-model.png",
        "p95_latency": charts_dir / "p95-latency-by-model.png",
        "time_to_first_token": charts_dir / "time-to-first-token-by-model.png",
        "tokens_per_second": charts_dir / "tokens-per-second-by-model.png",
        "json_success": charts_dir / "json-success-rate-by-model.png",
        "retry_rate": charts_dir / "retry-rate-by-model.png",
        "memory_usage": charts_dir / "memory-usage-by-model.png",
        "quality_vs_latency": charts_dir / "quality-vs-latency.png",
    }

    _bar_chart(analysis.summary, "average_latency_ms", "Average Latency (ms)", chart_paths["average_latency"])
    _bar_chart(analysis.summary, "p95_latency_ms", "P95 Latency (ms)", chart_paths["p95_latency"])
    _bar_chart(
        analysis.summary,
        "average_time_to_first_token_ms",
        "Average Time To First Token (ms)",
        chart_paths["time_to_first_token"],
    )
    _bar_chart(
        analysis.summary,
        "average_tokens_per_second",
        "Average Tokens Per Second",
        chart_paths["tokens_per_second"],
    )
    _bar_chart(analysis.summary, "json_validation_success_rate", "JSON Success Rate", chart_paths["json_success"])
    _bar_chart(analysis.summary, "retry_rate", "Retry Rate", chart_paths["retry_rate"])
    _bar_chart(analysis.summary, "average_memory_mb", "Average Memory (MB)", chart_paths["memory_usage"])
    _scatter_chart(analysis.summary, chart_paths["quality_vs_latency"])
    return chart_paths


def _bar_chart(summary, column: str, title: str, path: Path) -> None:
    labels = [f"{row.model}\ntemp={row.temperature:g}" for row in summary.itertuples()]
    values = summary[column].fillna(0)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _scatter_chart(summary, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(summary["average_latency_ms"], summary["quality_score"].fillna(0))
    for row in summary.itertuples():
        quality = 0 if pd.isna(row.quality_score) else row.quality_score
        plt.annotate(f"{row.model} t={row.temperature:g}", (row.average_latency_ms, quality))
    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Quality Score")
    plt.title("Quality vs Latency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _render_markdown(
    results_path: Path,
    analysis: AnalysisOutput,
    chart_paths: dict[str, Path],
    recommendations: list[ModelRecommendation],
    configured_scorer: str,
) -> str:
    generated_at = datetime.now(UTC).isoformat()
    summary_markdown = (
        _dataframe_to_markdown(analysis.summary)
        if not analysis.summary.empty
        else "No summary rows were available."
    )
    temperature_markdown = (
        _dataframe_to_markdown(analysis.temperature_comparison)
        if not analysis.temperature_comparison.empty
        else "No temperature comparison rows were available."
    )
    quality_dimension_markdown = (
        _dataframe_to_markdown(analysis.quality_dimensions)
        if not analysis.quality_dimensions.empty
        else "No quality dimension rows were available."
    )
    charts = "\n".join(f"![{name}]({path.as_posix()})" for name, path in chart_paths.items())
    return f"""# Local SLM Benchmark Report

Generated at: `{generated_at}`

Results source: `{results_path}`

## Executive Summary

This report compares local SLM benchmark runs across latency, throughput, structured JSON reliability, retry behavior, memory usage, output variance, temperature sensitivity, and quality scoring. Use the recommendation section for a practical model choice based on interactive speed, structured reliability, or balanced usage.

## Hardware And Environment

{_hardware_section(analysis)}

## Model List And Configuration

{_model_section(analysis)}

## Prompt Set Description

{_prompt_section(analysis)}

## Benchmark Methodology

The benchmark runner executes a configurable matrix of models, temperatures, prompts, and repeat counts on the benchmark machine. Each generation is validated against the required JSON schema, retried when invalid, persisted to JSONL, and paired with a metadata file describing the benchmark environment. Analysis on a separate machine loads the saved JSONL file, runs DeepDiff variance analysis, `{configured_scorer}` quality evaluation, and pandas aggregation to produce this report.

## Latency Comparison

Key latency metrics are summarized below. Lower average and p95 latency generally indicate a better interactive experience.

{summary_markdown}

## Throughput Comparison

Average tokens per second and time to first token are included in the aggregate table above and in the dedicated charts below.

## Memory Usage Comparison

Process memory and optional GPU memory usage were captured during benchmark runs on the benchmark machine.

## JSON Reliability Comparison

- JSON parse success rate: `{analysis.reliability.json_parse_success_rate:.1%}`
- Schema validation success rate: `{analysis.reliability.schema_validation_success_rate:.1%}`
- Retry rate: `{analysis.reliability.retry_rate:.1%}`
- Retry success rate: `{analysis.reliability.retry_success_rate:.1%}`
- Final failure rate: `{analysis.reliability.final_failure_rate:.1%}`
- Schema error categories: `{analysis.reliability.schema_error_categories}`

## Temperature And Determinism Analysis

This section compares repeat-run variance within each temperature and the distance between outputs at different temperatures for the same prompt.

{temperature_markdown}

## Quality Evaluation Summary

Quality scoring uses DeepEval with a local Ollama judge when enabled. The dimension table below breaks correctness, relevance, completeness, instruction following, and conciseness into separate scores.

{quality_dimension_markdown}

## Recommended Model Configuration

{_recommendation_section(recommendations)}

## Charts

{charts or "No charts were generated."}

## Interpretation Checklist

- Prefer models with low p95 latency and high tokens per second for interactive use.
- Prefer models with high JSON validation success and low retry rate for structured workflows.
- Compare quality score against latency before choosing the final local assistant model.
- Use variance and temperature comparison tables to understand determinism tradeoffs.

## Limitations And Next Steps

- Token counts prefer Ollama `eval_count` when available and fall back to a whitespace estimate otherwise.
- Quality scoring uses DeepEval with a local Ollama judge by default; runs fall back to lexical overlap if DeepEval is unavailable or fails.
- DeepEval dimension scoring is slower than lexical overlap because each result may require multiple judge-model calls.
- GPU metrics are included when `nvidia-smi` is available on the benchmark machine.
- Hardware, Ollama version, and model quantization can significantly change benchmark results.
"""


def _hardware_section(analysis: AnalysisOutput) -> str:
    metadata = analysis.metadata
    if metadata is None:
        return "No benchmark metadata file was found alongside the JSONL results. Copy the `.meta.json` file from the benchmark machine together with the JSONL file."
    lines = [
        f"- Platform: `{metadata.platform}`",
        f"- Python: `{metadata.python_version}`",
        f"- CPU count: `{metadata.cpu_count}`",
        f"- Total memory: `{metadata.total_memory_mb:.0f} MB`",
        f"- Ollama host: `{metadata.ollama_host}`",
        f"- Ollama version: `{metadata.ollama_version or 'unknown'}`",
        f"- Benchmark duration: `{metadata.benchmark_duration_ms or 0:.0f} ms`",
    ]
    if metadata.gpu_name:
        lines.append(f"- GPU: `{metadata.gpu_name}`")
    if metadata.gpu_memory_total_mb is not None:
        lines.append(f"- GPU memory total: `{metadata.gpu_memory_total_mb:.0f} MB`")
    return "\n".join(lines)


def _model_section(analysis: AnalysisOutput) -> str:
    metadata = analysis.metadata
    if metadata and metadata.model_details:
        rows = [
            "| Model | Label | Quantization |",
            "| --- | --- | --- |",
        ]
        for model in metadata.model_details:
            rows.append(
                f"| {model.get('name', '')} | {model.get('label') or ''} | {model.get('quantization') or 'default'} |"
            )
        temperature_text = ", ".join(f"{value:g}" for value in metadata.temperatures)
        return "\n".join(rows + [f"", f"Temperatures: `{temperature_text}`", f"Runs per prompt: `{metadata.runs_per_prompt}`"])
    if analysis.summary.empty:
        return "No model summary was available."
    models = sorted(analysis.summary["model"].unique())
    temperatures = ", ".join(f"{value:g}" for value in sorted(analysis.summary["temperature"].unique()))
    return f"Models: `{', '.join(models)}`\n\nTemperatures: `{temperatures}`"


def _prompt_section(analysis: AnalysisOutput) -> str:
    metadata = analysis.metadata
    if metadata:
        category_lines = [f"- `{name}`: {count}" for name, count in sorted(metadata.prompt_categories.items())]
        return (
            f"- Prompt count: `{metadata.prompt_count}`\n"
            f"- Categories:\n" + "\n".join(category_lines)
        )
    categories = analysis.prompt_summary.get("categories", {})
    category_lines = [f"- `{name}`: {count}" for name, count in sorted(categories.items())]
    return (
        f"- Prompt count: `{analysis.prompt_summary.get('prompt_count', 0)}`\n"
        f"- Categories:\n" + ("\n".join(category_lines) if category_lines else "- None")
    )


def _recommendation_section(recommendations: list[ModelRecommendation]) -> str:
    if not recommendations:
        return "No recommendation could be generated from the available summary data."
    lines = []
    for recommendation in recommendations:
        lines.append(
            f"- **{recommendation.profile.title()}**: `{recommendation.model}` at temperature `{recommendation.temperature:g}` "
            f"(score `{recommendation.score:.2f}`). {recommendation.rationale}"
        )
    return "\n".join(lines)


def _dataframe_to_markdown(dataframe) -> str:
    columns = list(dataframe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in dataframe.to_dict(orient="records"):
        values = [_format_cell(row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else str(value)
