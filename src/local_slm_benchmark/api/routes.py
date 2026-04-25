"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, Response

from local_slm_benchmark.config import load_models_config
from local_slm_benchmark.models.ollama_client import OllamaClient
from local_slm_benchmark.models.schemas import GenerateRequest
from local_slm_benchmark.observability.metrics import prometheus_payload, record_generation
from local_slm_benchmark.validation.parser import validate_assistant_response


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/generate")
def generate(request: GenerateRequest) -> dict:
    config = load_models_config()
    client = OllamaClient(config.ollama_host)
    response = client.generate(request)
    validation = validate_assistant_response(response.raw_response)
    record_generation(request.model, request.temperature, response.timing, validation.valid_json, retry_count=0)
    return {
        "model": response.model,
        "temperature": response.temperature,
        "raw_response": response.raw_response,
        "parsed_response": validation.parsed,
        "valid_json": validation.valid_json,
        "validation_errors": validation.errors,
        "timing": response.timing.model_dump(),
    }


@router.get("/metrics")
def metrics() -> Response:
    payload, content_type = prometheus_payload()
    return Response(content=payload, media_type=content_type)

