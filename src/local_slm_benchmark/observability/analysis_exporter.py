"""Prometheus export for post-run analysis metrics."""

from __future__ import annotations

import math

import pandas as pd
from prometheus_client import Gauge

from local_slm_benchmark.observability.metrics import label_temperature


QUALITY_SCORE = Gauge("slm_quality_score", "Average quality score.", ["model", "temperature"])
OUTPUT_VARIANCE_SCORE = Gauge(
    "slm_output_variance_score",
    "Average output variance score.",
    ["model", "temperature"],
)
JSON_SUCCESS_RATE = Gauge(
    "slm_json_validation_success_rate",
    "JSON validation success rate.",
    ["model", "temperature"],
)
RETRY_RATE = Gauge("slm_retry_rate", "Retry rate.", ["model", "temperature"])
AVERAGE_LATENCY = Gauge("slm_average_latency_ms", "Average latency in milliseconds.", ["model", "temperature"])
P95_LATENCY = Gauge("slm_p95_latency_ms", "P95 latency in milliseconds.", ["model", "temperature"])
AVERAGE_TOKENS_PER_SECOND = Gauge(
    "slm_average_tokens_per_second",
    "Average output tokens per second.",
    ["model", "temperature"],
)
AVERAGE_MEMORY_MB = Gauge("slm_average_memory_mb", "Average process memory in MB.", ["model", "temperature"])


def export_analysis_metrics(summary: pd.DataFrame) -> None:
    for row in summary.to_dict(orient="records"):
        model = str(row["model"])
        temperature = label_temperature(float(row["temperature"]))
        labels = (model, temperature)

        _set_if_number(AVERAGE_LATENCY.labels(*labels), row.get("average_latency_ms"))
        _set_if_number(P95_LATENCY.labels(*labels), row.get("p95_latency_ms"))
        _set_if_number(AVERAGE_TOKENS_PER_SECOND.labels(*labels), row.get("average_tokens_per_second"))
        _set_if_number(AVERAGE_MEMORY_MB.labels(*labels), row.get("average_memory_mb"))
        _set_if_number(JSON_SUCCESS_RATE.labels(*labels), row.get("json_validation_success_rate"))
        _set_if_number(RETRY_RATE.labels(*labels), row.get("retry_rate"))
        _set_if_number(QUALITY_SCORE.labels(*labels), row.get("quality_score"))
        _set_if_number(OUTPUT_VARIANCE_SCORE.labels(*labels), row.get("output_variance_score"))


def _set_if_number(gauge, value: object) -> None:
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isnan(number):
        return
    gauge.set(number)

