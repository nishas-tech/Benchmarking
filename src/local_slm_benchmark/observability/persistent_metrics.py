"""Persisted metric bridge for CLI-produced benchmark data.

Prometheus scrapes the FastAPI process, but benchmark runs usually happen in a
separate CLI process. This module stores CLI metrics on disk so the API process
can load and expose them from `/metrics`.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd

from local_slm_benchmark.config import project_path
from local_slm_benchmark.models.schemas import BenchmarkResult
from local_slm_benchmark.observability.metrics import label_temperature


METRICS_PATH = project_path("results/prometheus-metrics.json")


def record_persisted_benchmark_result(result: BenchmarkResult, path: Path = METRICS_PATH) -> None:
    payload = _read_payload(path)
    runtime = payload.setdefault("runtime", {})
    key = _runtime_key(result.model, result.temperature)
    row = runtime.setdefault(
        key,
        {
            "model": result.model,
            "temperature": label_temperature(result.temperature),
            "benchmark_runs_total": 0,
            "json_validation_failures_total": 0,
            "retries_total": 0,
            "latest_tokens_per_second": 0.0,
            "latest_generation_latency_ms": 0.0,
            "latest_time_to_first_token_ms": None,
            "latest_process_memory_mb": None,
        },
    )
    row["benchmark_runs_total"] += 1
    if not result.valid_json:
        row["json_validation_failures_total"] += 1
    row["retries_total"] += result.retry_count
    row["latest_tokens_per_second"] = result.tokens_per_second
    row["latest_generation_latency_ms"] = result.total_latency_ms
    row["latest_time_to_first_token_ms"] = result.time_to_first_token_ms
    row["latest_process_memory_mb"] = result.memory_mb
    _write_payload(path, payload)


def persist_analysis_summary(summary: pd.DataFrame, path: Path = METRICS_PATH) -> None:
    payload = _read_payload(path)
    payload["analysis"] = [
        {
            "model": str(row["model"]),
            "temperature": label_temperature(float(row["temperature"])),
            "average_latency_ms": _number_or_none(row.get("average_latency_ms")),
            "p95_latency_ms": _number_or_none(row.get("p95_latency_ms")),
            "average_tokens_per_second": _number_or_none(row.get("average_tokens_per_second")),
            "average_memory_mb": _number_or_none(row.get("average_memory_mb")),
            "json_validation_success_rate": _number_or_none(row.get("json_validation_success_rate")),
            "retry_rate": _number_or_none(row.get("retry_rate")),
            "quality_score": _number_or_none(row.get("quality_score")),
            "output_variance_score": _number_or_none(row.get("output_variance_score")),
        }
        for row in summary.to_dict(orient="records")
    ]
    _write_payload(path, payload)


def load_persisted_metrics(path: Path = METRICS_PATH) -> dict[str, Any]:
    return _read_payload(path)


def _runtime_key(model: str, temperature: float) -> str:
    return f"{model}|{label_temperature(temperature)}"


def _read_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runtime": {}, "analysis": []}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return {"runtime": {}, "analysis": []}
    if not isinstance(payload, dict):
        return {"runtime": {}, "analysis": []}
    payload.setdefault("runtime", {})
    payload.setdefault("analysis", [])
    return payload


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as file:
        json.dump(payload, file, indent=2)
        temp_path = Path(file.name)
    temp_path.replace(path)


def _number_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number

