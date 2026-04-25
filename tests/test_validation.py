from local_slm_benchmark.validation.parser import extract_json_text, validate_assistant_response
from local_slm_benchmark.validation.retry import build_retry_prompt


def test_extract_json_text_from_fenced_block() -> None:
    raw = '```json\n{"answer": "ok", "confidence": 1, "notes": []}\n```'

    assert extract_json_text(raw) == '{"answer": "ok", "confidence": 1, "notes": []}'


def test_validate_assistant_response_success() -> None:
    result = validate_assistant_response('{"answer": "ok", "confidence": 0.9, "notes": []}')

    assert result.valid_json is True
    assert result.parsed == {"answer": "ok", "confidence": 0.9, "notes": []}


def test_validate_assistant_response_failure() -> None:
    result = validate_assistant_response("not json")

    assert result.valid_json is False
    assert result.errors


def test_build_retry_prompt_includes_errors() -> None:
    prompt = build_retry_prompt("Say hi", "hello", ["missing answer"])

    assert "missing answer" in prompt
    assert "Return only valid JSON" in prompt

