from local_slm_benchmark.evaluation.diff import compute_temperature_comparisons, compute_variance_scores
from local_slm_benchmark.models.schemas import BenchmarkResult


def _result(model: str, temperature: float, prompt_id: str, answer: str, repeat_index: int) -> BenchmarkResult:
    return BenchmarkResult(
        prompt_id=prompt_id,
        prompt_category="test",
        model=model,
        temperature=temperature,
        repeat_index=repeat_index,
        raw_response=answer,
        parsed_response={"answer": answer, "confidence": 1.0, "notes": []},
        valid_json=True,
        total_latency_ms=100,
        output_tokens=5,
        tokens_per_second=50,
    )


def test_compute_variance_scores_within_repeat_group() -> None:
    results = [
        _result("m1", 0.0, "p1", "same", 1),
        _result("m1", 0.0, "p1", "same", 2),
    ]
    scores = compute_variance_scores(results)
    assert scores[results[0].run_id] == 0.0
    assert scores[results[1].run_id] == 0.0


def test_compute_temperature_comparisons_across_temperatures() -> None:
    results = [
        _result("m1", 0.0, "p1", "cold answer", 1),
        _result("m1", 0.7, "p1", "warm answer", 1),
    ]
    comparisons = compute_temperature_comparisons(results)
    assert len(comparisons) == 1
    assert comparisons[0].cross_temperature_distance > 0.0
