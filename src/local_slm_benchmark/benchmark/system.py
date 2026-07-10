"""System metric collection."""

from __future__ import annotations

import subprocess

import psutil

from local_slm_benchmark.models.schemas import SystemSnapshot


def capture_system_snapshot() -> SystemSnapshot:
    process = psutil.Process()
    memory_info = process.memory_info()
    gpu_name, gpu_memory_used_mb, gpu_utilization_percent = _gpu_snapshot()
    return SystemSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_mb=memory_info.rss / (1024 * 1024),
        memory_percent=process.memory_percent(),
        gpu_name=gpu_name,
        gpu_memory_used_mb=gpu_memory_used_mb,
        gpu_utilization_percent=gpu_utilization_percent,
    )


def _gpu_snapshot() -> tuple[str | None, float | None, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None, None, None
    if not output:
        return None, None, None
    first_line = output.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return first_line, None, None
    name, memory_used, utilization = parts
    try:
        return name, float(memory_used), float(utilization)
    except ValueError:
        return name, None, None
