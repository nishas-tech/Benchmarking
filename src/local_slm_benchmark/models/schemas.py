"""Pydantic schemas shared across generation, benchmarking, and reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    """Default structured output expected from the local assistant."""

    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class BenchmarkPrompt(BaseModel):
    id: str
    category: str
    prompt: str
    expected_schema: str = "assistant_response"
    reference_answer: str | None = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
    system_prompt: str | None = None
    format: str | None = "json"


class GenerationTiming(BaseModel):
    time_to_first_token_ms: float | None = None
    total_latency_ms: float
    output_tokens: int
    tokens_per_second: float


class GenerationResponse(BaseModel):
    model: str
    prompt: str
    temperature: float
    raw_response: str
    timing: GenerationTiming


class ValidationResult(BaseModel):
    valid_json: bool
    parsed: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class SystemSnapshot(BaseModel):
    cpu_percent: float | None = None
    memory_mb: float | None = None
    memory_percent: float | None = None


class BenchmarkResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_id: str
    prompt_category: str
    model: str
    temperature: float
    repeat_index: int
    raw_response: str
    parsed_response: dict[str, Any] | None = None
    reference_answer: str | None = None
    valid_json: bool
    validation_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    final_failure: str | None = None
    time_to_first_token_ms: float | None = None
    total_latency_ms: float
    output_tokens: int
    tokens_per_second: float
    cpu_percent: float | None = None
    memory_mb: float | None = None
    memory_percent: float | None = None
    quality_score: float | None = None
    variance_score: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def schema_instructions() -> str:
    """Return the JSON-only contract sent to models."""

    return (
        "Return only valid JSON. Do not include markdown fences or prose outside the JSON. "
        "The JSON object must match this schema: "
        '{"answer": "string", "confidence": 0.0, "notes": ["string"]}. '
        "Use confidence as a number between 0 and 1 when you can estimate it. "
        "Use notes for brief caveats or an empty list."
    )

