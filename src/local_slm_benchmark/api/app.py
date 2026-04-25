"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from local_slm_benchmark.api.routes import router
from local_slm_benchmark.observability.logging import configure_logging
from local_slm_benchmark.observability.tracing import configure_tracing


def create_app() -> FastAPI:
    configure_logging()
    configure_tracing()
    app = FastAPI(title="Local SLM Benchmarking Assistant", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()

