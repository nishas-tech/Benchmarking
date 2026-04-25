"""Correction prompt helpers for invalid structured outputs."""

from __future__ import annotations

from local_slm_benchmark.models.schemas import schema_instructions


def build_retry_prompt(original_prompt: str, invalid_response: str, errors: list[str]) -> str:
    error_text = "\n".join(f"- {error}" for error in errors) or "- Unknown validation error"
    return f"""The previous response did not pass validation.

Original task:
{original_prompt}

Validation errors:
{error_text}

Invalid response:
{invalid_response}

Please answer the original task again.
{schema_instructions()}
"""

