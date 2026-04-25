"""Prompt file loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from local_slm_benchmark.config import project_path
from local_slm_benchmark.models.schemas import BenchmarkPrompt


def load_prompts(path: str | Path) -> list[BenchmarkPrompt]:
    prompt_path = project_path(path)
    with prompt_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON array")
    return [BenchmarkPrompt.model_validate(item) for item in data]


def limit_prompts(prompts: list[BenchmarkPrompt], limit: int | None) -> list[BenchmarkPrompt]:
    if limit is None:
        return prompts
    return prompts[: max(0, limit)]

