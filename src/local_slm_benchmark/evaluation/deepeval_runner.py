"""Quality evaluation helpers using DeepEval or lexical fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from local_slm_benchmark.config import load_benchmark_config, load_models_config
from local_slm_benchmark.models.schemas import BenchmarkResult


WORD_RE = re.compile(r"\w+")

QUALITY_DIMENSION_CRITERIA = {
    "correctness": "Score how factually correct the answer is relative to the reference answer.",
    "relevance": "Score how relevant the answer is to the user prompt.",
    "completeness": "Score whether the answer covers the important points from the reference answer.",
    "instruction_following": "Score whether the answer follows the requested JSON-style assistant format and stays concise.",
    "conciseness": "Score whether the answer is concise and avoids unnecessary filler.",
}


@dataclass(frozen=True)
class QualityEvaluation:
    score: float | None
    scorer: str
    reason: str | None = None
    dimensions: dict[str, float] = field(default_factory=dict)
    dimension_reasons: dict[str, str] = field(default_factory=dict)


def score_result(
    result: BenchmarkResult,
    *,
    prompt_text: str | None = None,
    scorer: str | None = None,
) -> float | None:
    return evaluate_quality(result, prompt_text=prompt_text, scorer=scorer).score


def evaluate_quality(
    result: BenchmarkResult,
    *,
    prompt_text: str | None = None,
    scorer: str | None = None,
) -> QualityEvaluation:
    if not result.parsed_response:
        return QualityEvaluation(score=None, scorer=_selected_scorer(scorer))

    answer = str(result.parsed_response.get("answer", "")).strip()
    if not answer:
        return QualityEvaluation(score=None, scorer=_selected_scorer(scorer))

    selected_scorer = _selected_scorer(scorer)
    if selected_scorer == "deepeval" and result.reference_answer:
        if not deepeval_available():
            return _lexical_evaluation(answer, result, reason="DeepEval is not installed; fell back to lexical overlap.")
        try:
            return deepeval_score(result, prompt_text=prompt_text, answer=answer)
        except Exception as exc:
            return _lexical_evaluation(
                answer,
                result,
                reason=f"DeepEval scoring failed; fell back to lexical overlap. ({exc})",
            )

    if result.reference_answer:
        return _lexical_evaluation(answer, result)

    return QualityEvaluation(score=None, scorer=selected_scorer)


def deepeval_score(
    result: BenchmarkResult,
    *,
    prompt_text: str | None,
    answer: str,
) -> QualityEvaluation:
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=prompt_text or result.prompt_id,
        actual_output=answer,
        expected_output=result.reference_answer,
    )
    dimensions: dict[str, float] = {}
    dimension_reasons: dict[str, str] = {}
    for name in QUALITY_DIMENSION_CRITERIA:
        metric = _build_deepeval_metric(name)
        metric.measure(test_case)
        if metric.score is not None:
            dimensions[name] = float(metric.score)
        reason = getattr(metric, "reason", None)
        if reason:
            dimension_reasons[name] = str(reason)

    overall = _mean(dimensions.values()) if dimensions else None
    return QualityEvaluation(
        score=overall,
        scorer="deepeval",
        reason="Average of DeepEval dimension scores.",
        dimensions=dimensions,
        dimension_reasons=dimension_reasons,
    )


def lexical_overlap_score(answer: str, reference: str) -> float:
    answer_terms = set(_terms(answer))
    reference_terms = set(_terms(reference))
    if not reference_terms:
        return 0.0
    return len(answer_terms & reference_terms) / len(reference_terms)


def deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
    except Exception:
        return False
    return True


def reset_quality_metric_cache() -> None:
    _cached_deepeval_metric.cache_clear()


def _lexical_evaluation(answer: str, result: BenchmarkResult, reason: str | None = None) -> QualityEvaluation:
    overlap = lexical_overlap_score(answer, result.reference_answer or "")
    instruction = 1.0 if result.valid_schema else 0.0
    conciseness = _conciseness_score(answer, result.reference_answer or "")
    dimensions = {
        "correctness": overlap,
        "relevance": overlap,
        "completeness": overlap,
        "instruction_following": instruction,
        "conciseness": conciseness,
    }
    return QualityEvaluation(
        score=_mean(dimensions.values()),
        scorer="lexical",
        reason=reason,
        dimensions=dimensions,
    )


def _conciseness_score(answer: str, reference: str) -> float:
    answer_len = max(len(answer.split()), 1)
    reference_len = max(len(reference.split()), 1)
    ratio = answer_len / reference_len
    if ratio <= 1.0:
        return 1.0
    if ratio >= 2.5:
        return 0.0
    return max(0.0, 1.0 - ((ratio - 1.0) / 1.5))


def _selected_scorer(scorer: str | None) -> str:
    if scorer:
        return scorer
    return load_benchmark_config().quality_scorer


@lru_cache(maxsize=32)
def _cached_deepeval_metric(dimension: str, judge_model: str, base_url: str, temperature: float):
    from deepeval.metrics import GEval
    from deepeval.models import OllamaModel
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name=f"BenchmarkQuality_{dimension}",
        model=OllamaModel(
            model=judge_model,
            base_url=base_url,
            temperature=temperature,
        ),
        criteria=QUALITY_DIMENSION_CRITERIA[dimension],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
    )


def _build_deepeval_metric(dimension: str):
    models_config = load_models_config()
    benchmark_config = load_benchmark_config()
    judge_model = benchmark_config.judge_model
    if not judge_model:
        if not models_config.models:
            raise ValueError("No judge_model configured and no models listed in config/models.yaml.")
        judge_model = models_config.models[0].name
    return _cached_deepeval_metric(
        dimension=dimension,
        judge_model=judge_model,
        base_url=models_config.ollama_host,
        temperature=benchmark_config.judge_temperature,
    )


def _terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def _mean(values) -> float | None:
    items = [float(value) for value in values]
    if not items:
        return None
    return float(sum(items) / len(items))
