"""Markdown report and chart generation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from local_slm_benchmark.benchmark.results import latest_results_file
from local_slm_benchmark.config import load_benchmark_config, project_path
from local_slm_benchmark.observability.analysis_exporter import export_analysis_metrics
from local_slm_benchmark.reporting.analysis import AnalysisOutput, analyze_results


def generate_report(results_path: str | Path | None = None) -> Path:
    config = load_benchmark_config()
    selected_results = Path(results_path) if results_path else latest_results_file(config.results_dir)
    if selected_results is None:
        raise FileNotFoundError("No benchmark result files found.")

    analysis = analyze_results(selected_results)
    export_analysis_metrics(analysis.summary)

    reports_dir = project_path(config.reports_dir)
    charts_dir = reports_dir / "charts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = _generate_charts(analysis, charts_dir)
    report_path = reports_dir / f"model-comparison-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.md"
    report_path.write_text(_render_markdown(selected_results, analysis, chart_paths), encoding="utf-8")
    return report_path


def _generate_charts(analysis: AnalysisOutput, charts_dir: Path) -> dict[str, Path]:
    if analysis.summary.empty:
        return {}

    chart_paths = {
        "average_latency": charts_dir / "average-latency-by-model.png",
        "tokens_per_second": charts_dir / "tokens-per-second-by-model.png",
        "json_success": charts_dir / "json-success-rate-by-model.png",
        "quality_vs_latency": charts_dir / "quality-vs-latency.png",
    }

    _bar_chart(analysis.summary, "average_latency_ms", "Average Latency (ms)", chart_paths["average_latency"])
    _bar_chart(
        analysis.summary,
        "average_tokens_per_second",
        "Average Tokens Per Second",
        chart_paths["tokens_per_second"],
    )
    _bar_chart(analysis.summary, "json_validation_success_rate", "JSON Success Rate", chart_paths["json_success"])
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


def _render_markdown(results_path: Path, analysis: AnalysisOutput, chart_paths: dict[str, Path]) -> str:
    generated_at = datetime.now(UTC).isoformat()
    summary_markdown = (
        _dataframe_to_markdown(analysis.summary)
        if not analysis.summary.empty
        else "No summary rows were available."
    )
    charts = "\n".join(f"![{name}]({path.as_posix()})" for name, path in chart_paths.items())
    return f"""# Local SLM Benchmark Report

Generated at: `{generated_at}`

Results source: `{results_path}`

## Executive Summary

This report compares local SLM benchmark runs across latency, throughput, structured JSON reliability, retry behavior, memory usage, output variance, and quality scoring.

## Methodology

The benchmark runner executes a configurable matrix of models, temperatures, prompts, and repeat counts. Each generation is validated against the required JSON schema, retried when invalid, persisted to JSONL, analyzed with pandas, compared with DeepDiff, and scored with the local quality scorer.

## Aggregate Results

{summary_markdown}

## Charts

{charts or "No charts were generated."}

## Interpretation Checklist

- Prefer models with low p95 latency and high tokens per second for interactive use.
- Prefer models with high JSON validation success and low retry rate for structured workflows.
- Compare quality score against latency before choosing the final local assistant model.
- Use variance score to understand how temperature changes determinism.

## Limitations

- Token counts are approximated from whitespace-separated output unless model tokenizer integration is added later.
- Quality scoring is local and lightweight by default; DeepEval can be expanded with a stronger judge model later.
- Hardware, Ollama version, and model quantization can significantly change benchmark results.
"""


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

