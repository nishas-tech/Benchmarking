from local_slm_benchmark.models.schemas import BenchmarkResult
from local_slm_benchmark.reporting.analysis import results_to_dataframe, summarize_dataframe


def test_summarize_dataframe_groups_by_model_and_temperature() -> None:
    rows = [
        BenchmarkResult(
            prompt_id="p1",
            prompt_category="test",
            model="m1",
            temperature=0,
            repeat_index=1,
            raw_response="{}",
            parsed_response={"answer": "ok"},
            valid_json=True,
            total_latency_ms=100,
            output_tokens=10,
            tokens_per_second=50,
            memory_mb=100,
            quality_score=0.5,
            variance_score=0,
        ),
        BenchmarkResult(
            prompt_id="p2",
            prompt_category="test",
            model="m1",
            temperature=0,
            repeat_index=1,
            raw_response="{}",
            parsed_response={"answer": "ok"},
            valid_json=False,
            retry_count=1,
            total_latency_ms=300,
            output_tokens=10,
            tokens_per_second=25,
            memory_mb=200,
            quality_score=1.0,
            variance_score=2,
        ),
    ]

    summary = summarize_dataframe(results_to_dataframe(rows))

    assert summary.loc[0, "average_latency_ms"] == 200
    assert summary.loc[0, "json_validation_success_rate"] == 0.5
    assert summary.loc[0, "retry_rate"] == 0.5

