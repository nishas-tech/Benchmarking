from pathlib import Path

import pandas as pd

from local_slm_benchmark.benchmark.metadata import BenchmarkRunMetadata, write_run_metadata
from local_slm_benchmark.models.schemas import BenchmarkResult
from local_slm_benchmark.reporting.analysis import AnalysisOutput, ReliabilitySummary
from local_slm_benchmark.reporting.recommendation import ModelRecommendation
from local_slm_benchmark.reporting.report import _render_markdown


def test_render_markdown_includes_recommendation_and_hardware(tmp_path: Path) -> None:
    results_path = tmp_path / "benchmark-test.jsonl"
    metadata = BenchmarkRunMetadata(
        run_file=results_path.name,
        platform="test-platform",
        python_version="3.13.0",
        cpu_count=8,
        total_memory_mb=16000,
        ollama_host="http://localhost:11434",
        models=["m1"],
        model_details=[{"name": "m1", "label": "Model 1", "quantization": None}],
        temperatures=[0.0],
        prompt_count=2,
        prompt_categories={"summarization": 2},
        runs_per_prompt=1,
        case_count=2,
        benchmark_duration_ms=5000,
    )
    write_run_metadata(metadata, results_path)

    summary = pd.DataFrame(
        [
            {
                "model": "m1",
                "temperature": 0.0,
                "average_latency_ms": 100,
                "average_tokens_per_second": 30,
                "quality_score": 0.8,
                "json_validation_success_rate": 1.0,
                "retry_rate": 0.0,
                "final_failure_rate": 0.0,
                "output_variance_score": 0.0,
            }
        ]
    )
    analysis = AnalysisOutput(
        results=[
            BenchmarkResult(
                prompt_id="p1",
                prompt_category="summarization",
                model="m1",
                temperature=0.0,
                repeat_index=1,
                raw_response="{}",
                valid_json=True,
                total_latency_ms=100,
                output_tokens=5,
                tokens_per_second=30,
            )
        ],
        dataframe=summary,
        summary=summary,
        reliability=ReliabilitySummary(1.0, 1.0, 0.0, 0.0, 0.0, {}),
        temperature_comparison=pd.DataFrame(),
        quality_dimensions=pd.DataFrame(),
        metadata=metadata,
        prompt_summary={"prompt_count": 2, "categories": {"summarization": 2}},
    )
    recommendations = [
        ModelRecommendation(
            profile="balanced",
            model="m1",
            temperature=0.0,
            score=0.91,
            rationale="Best overall tradeoff.",
        )
    ]

    markdown = _render_markdown(results_path, analysis, {}, recommendations, "lexical")
    assert "## Recommended Model Configuration" in markdown
    assert "test-platform" in markdown
    assert "Best overall tradeoff." in markdown
