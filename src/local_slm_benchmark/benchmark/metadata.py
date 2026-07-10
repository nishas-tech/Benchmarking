"""Benchmark run metadata captured on the benchmark machine."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil
from pydantic import BaseModel, Field

from local_slm_benchmark.config import BenchmarkConfig, ModelsConfig, project_path
from local_slm_benchmark.models.schemas import BenchmarkPrompt


class BenchmarkRunMetadata(BaseModel):
    run_file: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    benchmark_duration_ms: float | None = None
    python_version: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=platform.platform)
    cpu_count: int = Field(default_factory=lambda: psutil.cpu_count(logical=True) or 0)
    total_memory_mb: float = Field(default_factory=lambda: psutil.virtual_memory().total / (1024 * 1024))
    ollama_host: str = "http://localhost:11434"
    ollama_version: str | None = None
    models: list[str] = Field(default_factory=list)
    model_details: list[dict[str, Any]] = Field(default_factory=list)
    temperatures: list[float] = Field(default_factory=list)
    prompt_count: int = 0
    prompt_categories: dict[str, int] = Field(default_factory=dict)
    runs_per_prompt: int = 1
    case_count: int = 0
    gpu_name: str | None = None
    gpu_memory_total_mb: float | None = None


def metadata_path_for(results_path: str | Path) -> Path:
    path = project_path(results_path)
    return path.with_suffix(".meta.json")


def capture_run_metadata(
    *,
    results_path: Path,
    models_config: ModelsConfig,
    benchmark_config: BenchmarkConfig,
    prompts: list[BenchmarkPrompt],
    case_count: int,
    runs_per_prompt: int,
    selected_models: list[str],
) -> BenchmarkRunMetadata:
    categories: dict[str, int] = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1

    model_details = [
        {
            "name": model.name,
            "label": model.label,
            "quantization": model.quantization,
        }
        for model in models_config.models
        if model.name in selected_models
    ]
    gpu_name, gpu_memory_total_mb = _detect_gpu()

    return BenchmarkRunMetadata(
        run_file=results_path.name,
        ollama_host=models_config.ollama_host,
        ollama_version=_fetch_ollama_version(models_config.ollama_host),
        models=selected_models,
        model_details=model_details,
        temperatures=list(benchmark_config.temperatures),
        prompt_count=len(prompts),
        prompt_categories=categories,
        runs_per_prompt=runs_per_prompt,
        case_count=case_count,
        gpu_name=gpu_name,
        gpu_memory_total_mb=gpu_memory_total_mb,
    )


def write_run_metadata(metadata: BenchmarkRunMetadata, results_path: str | Path) -> Path:
    path = metadata_path_for(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run_metadata(results_path: str | Path) -> BenchmarkRunMetadata | None:
    path = metadata_path_for(results_path)
    if not path.exists():
        return None
    try:
        return BenchmarkRunMetadata.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def finalize_run_metadata(
    results_path: str | Path,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> BenchmarkRunMetadata | None:
    metadata = load_run_metadata(results_path)
    if metadata is None:
        return None
    duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000)
    updated = metadata.model_copy(
        update={
            "finished_at": finished_at,
            "benchmark_duration_ms": duration_ms,
        }
    )
    write_run_metadata(updated, results_path)
    return updated


def _fetch_ollama_version(host: str) -> str | None:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{host.rstrip('/')}/api/version")
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None
    version = payload.get("version")
    return str(version) if version else None


def _detect_gpu() -> tuple[str | None, float | None]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None, None
    if not output:
        return None, None
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 2:
        return first_line, None
    name, memory = parts
    try:
        return name, float(memory)
    except ValueError:
        return name, None
