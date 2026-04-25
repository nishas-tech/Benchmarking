"""JSON parsing and schema validation for model responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from local_slm_benchmark.models.schemas import AssistantResponse, ValidationResult


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_text(raw_response: str) -> str:
    """Extract a JSON object from common LLM response formats."""

    text = raw_response.strip()
    fenced = FENCED_JSON_RE.search(text)
    if fenced:
        text = fenced.group("body").strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def parse_json_response(raw_response: str) -> dict[str, Any]:
    return json.loads(extract_json_text(raw_response))


def validate_assistant_response(raw_response: str) -> ValidationResult:
    try:
        parsed = parse_json_response(raw_response)
    except json.JSONDecodeError as exc:
        return ValidationResult(valid_json=False, errors=[f"JSON parse error: {exc}"])

    try:
        validated = AssistantResponse.model_validate(parsed)
    except ValidationError as exc:
        return ValidationResult(valid_json=False, parsed=parsed, errors=[str(exc)])

    return ValidationResult(valid_json=True, parsed=validated.model_dump(), errors=[])

