from local_slm_benchmark.evaluation.deepeval_runner import (
    QualityEvaluation,
    deepeval_score,
    evaluate_quality,
    lexical_overlap_score,
    reset_quality_metric_cache,
)
from local_slm_benchmark.models.schemas import BenchmarkResult


def _sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        prompt_id="summarize_001",
        prompt_category="summarization",
        model="llama3.2:3b",
        temperature=0.0,
        repeat_index=1,
        raw_response='{"answer": "privacy latency hardware", "confidence": 1.0, "notes": []}',
        parsed_response={"answer": "privacy latency hardware", "confidence": 1.0, "notes": []},
        reference_answer="Local SLMs can improve privacy, reduce network latency, and allow experimentation on local hardware.",
        valid_json=True,
        valid_schema=True,
        valid_json_parse=True,
        total_latency_ms=100,
        output_tokens=10,
        tokens_per_second=50,
    )


def test_lexical_overlap_score_counts_shared_terms() -> None:
    score = lexical_overlap_score(
        "privacy latency hardware",
        "Local SLMs can improve privacy, reduce network latency, and allow experimentation on local hardware.",
    )
    assert score > 0.2


def test_evaluate_quality_uses_lexical_scorer_when_requested() -> None:
    evaluation = evaluate_quality(_sample_result(), scorer="lexical")
    assert evaluation.scorer == "lexical"
    assert evaluation.score is not None
    assert evaluation.dimensions
    assert 0.0 <= evaluation.score <= 1.0


def test_evaluate_quality_returns_none_without_reference_answer() -> None:
    result = _sample_result().model_copy(update={"reference_answer": None})
    evaluation = evaluate_quality(result, scorer="lexical")
    assert evaluation.score is None


def test_deepeval_score_returns_dimension_scores(monkeypatch) -> None:
    class FakeMetric:
        def __init__(self, score: float) -> None:
            self.score = score
            self.reason = "Strong match to the reference answer."

        def measure(self, test_case) -> None:
            assert test_case.input == "Summarize the benefits of local SLMs."
            assert test_case.actual_output == "privacy latency hardware"
            assert "privacy" in test_case.expected_output

    monkeypatch.setattr(
        "local_slm_benchmark.evaluation.deepeval_runner._build_deepeval_metric",
        lambda dimension: FakeMetric(0.9 if dimension == "correctness" else 0.8),
    )

    evaluation = deepeval_score(
        _sample_result(),
        prompt_text="Summarize the benefits of local SLMs.",
        answer="privacy latency hardware",
    )
    assert evaluation.scorer == "deepeval"
    assert evaluation.score is not None
    assert evaluation.dimensions["correctness"] == 0.9
    assert evaluation.dimensions["relevance"] == 0.8


def test_evaluate_quality_falls_back_to_lexical_when_deepeval_fails(monkeypatch) -> None:
    def _raise_error(*args, **kwargs):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr("local_slm_benchmark.evaluation.deepeval_runner.deepeval_available", lambda: True)
    monkeypatch.setattr("local_slm_benchmark.evaluation.deepeval_runner.deepeval_score", _raise_error)

    evaluation = evaluate_quality(_sample_result(), scorer="deepeval")
    assert evaluation.scorer == "lexical"
    assert evaluation.score is not None
    assert evaluation.reason is not None
    assert "DeepEval scoring failed" in evaluation.reason


def test_evaluate_quality_falls_back_when_deepeval_not_installed(monkeypatch) -> None:
    monkeypatch.setattr("local_slm_benchmark.evaluation.deepeval_runner.deepeval_available", lambda: False)

    evaluation = evaluate_quality(_sample_result(), scorer="deepeval")
    assert evaluation.scorer == "lexical"
    assert evaluation.score is not None
    assert evaluation.reason is not None
    assert "not installed" in evaluation.reason


def test_reset_quality_metric_cache_clears_cached_metric() -> None:
    reset_quality_metric_cache()
