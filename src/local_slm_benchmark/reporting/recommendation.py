"""Model recommendation helpers for benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelRecommendation:
    profile: str
    model: str
    temperature: float
    score: float
    rationale: str


def recommend_models(summary: pd.DataFrame) -> list[ModelRecommendation]:
    if summary.empty:
        return []

    normalized = _normalize_summary(summary)
    recommendations = [
        _best_row(normalized, "interactive", {"latency": 0.45, "throughput": 0.25, "quality": 0.15, "reliability": 0.15}),
        _best_row(normalized, "structured", {"latency": 0.15, "throughput": 0.10, "quality": 0.20, "reliability": 0.55}),
        _best_row(normalized, "balanced", {"latency": 0.25, "throughput": 0.20, "quality": 0.30, "reliability": 0.25}),
    ]
    return [item for item in recommendations if item is not None]


def _normalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["latency_score"] = _inverse_normalize(frame["average_latency_ms"])
    frame["throughput_score"] = _normalize(frame["average_tokens_per_second"])
    frame["quality_score_norm"] = _normalize(frame["quality_score"].fillna(0))
    frame["reliability_score"] = _normalize(frame["json_validation_success_rate"]) * 0.5
    frame["reliability_score"] += _normalize(1 - frame["retry_rate"].fillna(0)) * 0.3
    frame["reliability_score"] += _normalize(1 - frame["final_failure_rate"].fillna(0)) * 0.2
    frame["variance_penalty"] = _normalize(frame["output_variance_score"].fillna(0))
    return frame


def _best_row(frame: pd.DataFrame, profile: str, weights: dict[str, float]) -> ModelRecommendation | None:
    if frame.empty:
        return None
    scores = (
        frame["latency_score"] * weights["latency"]
        + frame["throughput_score"] * weights["throughput"]
        + frame["quality_score_norm"] * weights["quality"]
        + frame["reliability_score"] * weights["reliability"]
        - frame["variance_penalty"] * 0.05
    )
    best_index = scores.idxmax()
    row = frame.loc[best_index]
    return ModelRecommendation(
        profile=profile,
        model=str(row["model"]),
        temperature=float(row["temperature"]),
        score=float(scores.loc[best_index]),
        rationale=_rationale(profile, row),
    )


def _rationale(profile: str, row: pd.Series) -> str:
    if profile == "interactive":
        return (
            f"Low average latency ({row['average_latency_ms']:.0f} ms) and strong throughput "
            f"({row['average_tokens_per_second']:.1f} tok/s) make this the best fit for responsive local use."
        )
    if profile == "structured":
        return (
            f"High JSON success ({row['json_validation_success_rate']:.1%}) and lower retry/failure pressure "
            f"make this the safest choice for structured assistant workflows."
        )
    return (
        f"Balanced latency ({row['average_latency_ms']:.0f} ms), quality ({row.get('quality_score', 0):.2f}), "
        f"and reliability ({row['json_validation_success_rate']:.1%}) provide the best overall tradeoff."
    )


def _normalize(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum == minimum:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - minimum) / (maximum - minimum)


def _inverse_normalize(series: pd.Series) -> pd.Series:
    minimum = float(series.min())
    maximum = float(series.max())
    if maximum == minimum:
        return pd.Series([1.0] * len(series), index=series.index)
    return 1 - ((series - minimum) / (maximum - minimum))
