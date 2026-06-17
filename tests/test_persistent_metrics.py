import pandas as pd

from local_slm_benchmark.models.schemas import BenchmarkResult
from local_slm_benchmark.observability.persistent_metrics import (
    load_persisted_metrics,
    persist_analysis_summary,
    record_persisted_benchmark_result,
)


def test_persistent_metrics_bridge_records_runtime_and_analysis(tmp_path) -> None:
    path = tmp_path / "prometheus-metrics.json"
    result = BenchmarkResult(
        prompt_id="p1",
        prompt_category="test",
        model="llama3.2:3b",
        temperature=0.7,
        repeat_index=1,
        raw_response='{"answer": "ok"}',
        parsed_response={"answer": "ok"},
        valid_json=True,
        retry_count=1,
        total_latency_ms=123,
        output_tokens=10,
        tokens_per_second=20,
        memory_mb=100,
    )

    record_persisted_benchmark_result(result, path=path)
    persist_analysis_summary(
        pd.DataFrame(
            [
                {
                    "model": "llama3.2:3b",
                    "temperature": 0.7,
                    "average_latency_ms": 123,
                    "p95_latency_ms": 123,
                    "average_tokens_per_second": 20,
                    "average_memory_mb": 100,
                    "json_validation_success_rate": 1,
                    "retry_rate": 1,
                    "quality_score": 0.5,
                    "output_variance_score": 0,
                }
            ]
        ),
        path=path,
    )

    payload = load_persisted_metrics(path=path)

    runtime = payload["runtime"]["llama3.2:3b|0.7"]
    assert runtime["benchmark_runs_total"] == 1
    assert runtime["retries_total"] == 1
    assert payload["analysis"][0]["quality_score"] == 0.5

