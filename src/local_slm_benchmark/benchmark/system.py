"""System metric collection."""

from __future__ import annotations

import psutil

from local_slm_benchmark.models.schemas import SystemSnapshot


def capture_system_snapshot() -> SystemSnapshot:
    process = psutil.Process()
    memory_info = process.memory_info()
    return SystemSnapshot(
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_mb=memory_info.rss / (1024 * 1024),
        memory_percent=process.memory_percent(),
    )

