"""Quality evaluation helpers.

The default scorer is local and deterministic so the project can run without a
cloud judge. DeepEval can be integrated later for richer LLM-as-judge scoring.
"""

from __future__ import annotations

import re

from local_slm_benchmark.models.schemas import BenchmarkResult


WORD_RE = re.compile(r"\w+")


def score_result(result: BenchmarkResult) -> float | None:
    if not result.reference_answer or not result.parsed_response:
        return None
    answer = str(result.parsed_response.get("answer", ""))
    return lexical_overlap_score(answer, result.reference_answer)


def lexical_overlap_score(answer: str, reference: str) -> float:
    answer_terms = set(_terms(answer))
    reference_terms = set(_terms(reference))
    if not reference_terms:
        return 0.0
    return len(answer_terms & reference_terms) / len(reference_terms)


def _terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def deepeval_available() -> bool:
    try:
        import deepeval  # noqa: F401
    except Exception:
        return False
    return True

