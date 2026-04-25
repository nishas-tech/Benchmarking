from local_slm_benchmark.benchmark.runner import BenchmarkCase, BenchmarkRunner
from local_slm_benchmark.config import BenchmarkConfig, ModelConfig, ModelsConfig
from local_slm_benchmark.models.schemas import BenchmarkPrompt, GenerateRequest, GenerationResponse, GenerationTiming


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[GenerateRequest] = []

    def generate(self, request: GenerateRequest) -> GenerationResponse:
        self.calls.append(request)
        raw = self.responses.pop(0)
        return GenerationResponse(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            raw_response=raw,
            timing=GenerationTiming(
                time_to_first_token_ms=10,
                total_latency_ms=100,
                output_tokens=5,
                tokens_per_second=50,
            ),
        )


def test_runner_retries_invalid_json() -> None:
    runner = BenchmarkRunner(
        models_config=ModelsConfig(models=[ModelConfig(name="test-model")]),
        benchmark_config=BenchmarkConfig(max_retries=1),
        client=FakeClient(["not json", '{"answer": "ok", "confidence": 1, "notes": []}']),
    )
    case = BenchmarkCase(
        prompt=BenchmarkPrompt(id="p1", category="test", prompt="Do work"),
        model="test-model",
        temperature=0,
        repeat_index=1,
    )

    result = runner.run_case(case)

    assert result.valid_json is True
    assert result.retry_count == 1
    assert result.total_latency_ms == 200
    assert result.parsed_response == {"answer": "ok", "confidence": 1.0, "notes": []}

