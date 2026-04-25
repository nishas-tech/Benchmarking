"""Benchmark result persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from local_slm_benchmark.config import project_path
from local_slm_benchmark.models.schemas import BenchmarkResult


def new_results_path(results_dir: str | Path) -> Path:
    directory = project_path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return directory / f"benchmark-{stamp}.jsonl"


def append_result(path: str | Path, result: BenchmarkResult) -> None:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(result.model_dump_json() + "\n")


def load_results(path: str | Path) -> list[BenchmarkResult]:
    result_path = project_path(path)
    rows: list[BenchmarkResult] = []
    with result_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(BenchmarkResult.model_validate(json.loads(line)))
    return rows


def latest_results_file(results_dir: str | Path = "results") -> Path | None:
    directory = project_path(results_dir)
    if not directory.exists():
        return None
    files = sorted(directory.glob("benchmark-*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None

