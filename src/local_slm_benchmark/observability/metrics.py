"""Prometheus runtime metrics."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from local_slm_benchmark.models.schemas import BenchmarkResult, GenerationTiming


GENERATION_LATENCY = Histogram(
    "slm_generation_latency_ms",
    "Total generation latency in milliseconds.",
    ["model", "temperature"],
)
TIME_TO_FIRST_TOKEN = Histogram(
    "slm_time_to_first_token_ms",
    "Time to first streamed token in milliseconds.",
    ["model", "temperature"],
)
TOKENS_PER_SECOND = Gauge(
    "slm_tokens_per_second",
    "Approximate output tokens per second.",
    ["model", "temperature"],
)
JSON_VALIDATION_FAILURES = Counter(
    "slm_json_validation_failures_total",
    "Total JSON validation failures.",
    ["model", "temperature"],
)
RETRIES = Counter(
    "slm_retries_total",
    "Total structured-output retries.",
    ["model", "temperature"],
)
BENCHMARK_RUNS = Counter(
    "slm_benchmark_runs_total",
    "Total benchmark cases completed.",
    ["model", "temperature", "valid_json"],
)
PROCESS_MEMORY_MB = Gauge(
    "slm_process_memory_mb",
    "Benchmark process resident memory in MB.",
    ["model", "temperature"],
)


def label_temperature(temperature: float) -> str:
    return f"{temperature:g}"


def record_generation(model: str, temperature: float, timing: GenerationTiming, valid_json: bool, retry_count: int) -> None:
    labels = (model, label_temperature(temperature))
    GENERATION_LATENCY.labels(*labels).observe(timing.total_latency_ms)
    if timing.time_to_first_token_ms is not None:
        TIME_TO_FIRST_TOKEN.labels(*labels).observe(timing.time_to_first_token_ms)
    TOKENS_PER_SECOND.labels(*labels).set(timing.tokens_per_second)
    if not valid_json:
        JSON_VALIDATION_FAILURES.labels(*labels).inc()
    if retry_count:
        RETRIES.labels(*labels).inc(retry_count)
    BENCHMARK_RUNS.labels(model, label_temperature(temperature), str(valid_json).lower()).inc()


def record_benchmark_result(result: BenchmarkResult) -> None:
    timing = GenerationTiming(
        time_to_first_token_ms=result.time_to_first_token_ms,
        total_latency_ms=result.total_latency_ms,
        output_tokens=result.output_tokens,
        tokens_per_second=result.tokens_per_second,
    )
    record_generation(result.model, result.temperature, timing, result.valid_json, result.retry_count)
    if result.memory_mb is not None:
        PROCESS_MEMORY_MB.labels(result.model, label_temperature(result.temperature)).set(result.memory_mb)


def prometheus_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST

