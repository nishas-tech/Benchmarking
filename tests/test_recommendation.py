import pandas as pd

from local_slm_benchmark.reporting.recommendation import recommend_models


def test_recommend_models_returns_three_profiles() -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "fast-model",
                "temperature": 0.0,
                "average_latency_ms": 100,
                "average_tokens_per_second": 40,
                "quality_score": 0.7,
                "json_validation_success_rate": 0.9,
                "retry_rate": 0.1,
                "final_failure_rate": 0.05,
                "output_variance_score": 0.2,
            },
            {
                "model": "quality-model",
                "temperature": 0.0,
                "average_latency_ms": 400,
                "average_tokens_per_second": 15,
                "quality_score": 0.95,
                "json_validation_success_rate": 0.99,
                "retry_rate": 0.01,
                "final_failure_rate": 0.01,
                "output_variance_score": 0.1,
            },
        ]
    )

    recommendations = recommend_models(summary)
    assert len(recommendations) == 3
    assert {item.profile for item in recommendations} == {"interactive", "structured", "balanced"}
