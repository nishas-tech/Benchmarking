"""Configuration loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelConfig(BaseModel):
    name: str
    label: str | None = None
    quantization: str | None = None


class ModelsConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    models: list[ModelConfig] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    temperatures: list[float] = Field(default_factory=lambda: [0.0])
    runs_per_prompt: int = 1
    max_retries: int = 1
    prompts_path: str = "prompts/benchmark_prompts.json"
    results_dir: str = "results"
    reports_dir: str = "reports"
    default_response_schema: str = "assistant_response"
    quality_scorer: str = "deepeval"
    judge_model: str | None = None
    judge_temperature: float = 0.0


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_yaml(path: str | Path) -> dict[str, Any]:
    with project_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_models_config(path: str | Path = "config/models.yaml") -> ModelsConfig:
    config = ModelsConfig.model_validate(load_yaml(path))
    host_override = os.getenv("OLLAMA_HOST")
    if host_override:
        config.ollama_host = host_override
    return config


def load_benchmark_config(path: str | Path = "config/benchmark.yaml") -> BenchmarkConfig:
    config = BenchmarkConfig.model_validate(load_yaml(path))
    if results_dir := os.getenv("SLM_RESULTS_DIR"):
        config.results_dir = results_dir
    if reports_dir := os.getenv("SLM_REPORTS_DIR"):
        config.reports_dir = reports_dir
    if max_retries := os.getenv("SLM_MAX_RETRIES"):
        config.max_retries = int(max_retries)
    if quality_scorer := os.getenv("SLM_QUALITY_SCORER"):
        config.quality_scorer = quality_scorer
    if judge_model := os.getenv("SLM_JUDGE_MODEL"):
        config.judge_model = judge_model
    if judge_temperature := os.getenv("SLM_JUDGE_TEMPERATURE"):
        config.judge_temperature = float(judge_temperature)
    return config


def model_quantization(models_config: ModelsConfig, model_name: str) -> str | None:
    for model in models_config.models:
        if model.name == model_name:
            return model.quantization
    return None

