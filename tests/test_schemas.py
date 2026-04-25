from local_slm_benchmark.models.schemas import AssistantResponse, BenchmarkPrompt, schema_instructions


def test_assistant_response_schema_accepts_expected_shape() -> None:
    response = AssistantResponse.model_validate({"answer": "done", "confidence": 0.5, "notes": ["brief"]})

    assert response.answer == "done"
    assert response.confidence == 0.5
    assert response.notes == ["brief"]


def test_benchmark_prompt_defaults_schema() -> None:
    prompt = BenchmarkPrompt(id="p1", category="test", prompt="Return JSON")

    assert prompt.expected_schema == "assistant_response"


def test_schema_instructions_define_json_contract() -> None:
    instructions = schema_instructions()

    assert "Return only valid JSON" in instructions
    assert "answer" in instructions

