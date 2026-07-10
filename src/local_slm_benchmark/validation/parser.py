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


def classify_validation_error(errors: list[str]) -> str | None:
    if not errors:
        return None
    joined = " ".join(errors).lower()
    if "json parse error" in joined:
        return "json_parse"
    if "field required" in joined or "missing" in joined:
        return "missing_field"
    if "type" in joined or "input should be" in joined or "value_error" in joined:
        return "type_mismatch"
    if "validation error" in joined:
        return "schema_validation"
    return "other"


def validate_assistant_response(raw_response: str) -> ValidationResult:
    try:
        parsed = parse_json_response(raw_response)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            valid_json=False,
            valid_json_parse=False,
            valid_schema=False,
            error_category="json_parse",
            errors=[f"JSON parse error: {exc}"],
        )

    try:
        validated = AssistantResponse.model_validate(parsed)
    except ValidationError as exc:
        errors = [str(exc)]
        return ValidationResult(
            valid_json=False,
            valid_json_parse=True,
            valid_schema=False,
            error_category=classify_validation_error(errors),
            parsed=parsed,
            errors=errors,
        )

    return ValidationResult(
        valid_json=True,
        valid_json_parse=True,
        valid_schema=True,
        error_category=None,
        parsed=validated.model_dump(),
        errors=[],
    )
