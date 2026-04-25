"""Benchmark matrix runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from local_slm_benchmark.benchmark.results import append_result, new_results_path
from local_slm_benchmark.benchmark.system import capture_system_snapshot
from local_slm_benchmark.config import BenchmarkConfig, ModelsConfig, load_benchmark_config, load_models_config
from local_slm_benchmark.models.ollama_client import OllamaClient, OllamaGenerationError
from local_slm_benchmark.models.schemas import BenchmarkPrompt, BenchmarkResult, GenerateRequest, GenerationResponse
from local_slm_benchmark.observability.logging import get_logger
from local_slm_benchmark.observability.metrics import record_benchmark_result
from local_slm_benchmark.observability.tracing import get_tracer
from local_slm_benchmark.prompts import limit_prompts, load_prompts
from local_slm_benchmark.validation.parser import validate_assistant_response
from local_slm_benchmark.validation.retry import build_retry_prompt


console = Console()
logger = get_logger(__name__)
tracer = get_tracer(__name__)


@dataclass(frozen=True)
class BenchmarkCase:
    prompt: BenchmarkPrompt
    model: str
    temperature: float
    repeat_index: int


class BenchmarkRunner:
    def __init__(
        self,
        models_config: ModelsConfig | None = None,
        benchmark_config: BenchmarkConfig | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self.models_config = models_config or load_models_config()
        self.benchmark_config = benchmark_config or load_benchmark_config()
        self.client = client or OllamaClient(self.models_config.ollama_host)

    def run(
        self,
        limit: int | None = None,
        runs_per_prompt: int | None = None,
        models: list[str] | None = None,
    ) -> Path:
        prompts = limit_prompts(load_prompts(self.benchmark_config.prompts_path), limit)
        cases = self._build_cases(prompts, runs_per_prompt or self.benchmark_config.runs_per_prompt, models)
        output_path = new_results_path(self.benchmark_config.results_dir)

        with Progress() as progress:
            task = progress.add_task("Running benchmark", total=len(cases))
            for case in cases:
                result = self.run_case(case)
                append_result(output_path, result)
                progress.update(task, advance=1)

        self._print_summary(output_path, len(cases))
        return output_path

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        with tracer.start_as_current_span("benchmark.run_case") as span:
            span.set_attribute("model.name", case.model)
            span.set_attribute("model.temperature", case.temperature)
            span.set_attribute("prompt.id", case.prompt.id)
            request = GenerateRequest(model=case.model, prompt=case.prompt.prompt, temperature=case.temperature)
            try:
                generation = self.client.generate(request)
            except OllamaGenerationError as exc:
                result = self._error_result(case, exc)
                record_benchmark_result(result)
                span.set_attribute("response.valid_json", False)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(exc))
                logger.warning(
                    "benchmark_case_failed",
                    model=result.model,
                    temperature=result.temperature,
                    prompt_id=result.prompt_id,
                    error=result.final_failure,
                )
                return result
            validation = validate_assistant_response(generation.raw_response)
            retry_count = 0
            final_failure: str | None = None
            total_latency_ms = generation.timing.total_latency_ms
            final_generation = generation

            while not validation.valid_json and retry_count < self.benchmark_config.max_retries:
                retry_count += 1
                retry_prompt = build_retry_prompt(case.prompt.prompt, final_generation.raw_response, validation.errors)
                try:
                    final_generation = self.client.generate(
                        GenerateRequest(model=case.model, prompt=retry_prompt, temperature=case.temperature)
                    )
                except OllamaGenerationError as exc:
                    result = self._error_result(case, exc, retry_count=retry_count, elapsed_ms=total_latency_ms + exc.elapsed_ms)
                    record_benchmark_result(result)
                    span.set_attribute("response.valid_json", False)
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(exc))
                    logger.warning(
                        "benchmark_case_failed",
                        model=result.model,
                        temperature=result.temperature,
                        prompt_id=result.prompt_id,
                        error=result.final_failure,
                    )
                    return result
                total_latency_ms += final_generation.timing.total_latency_ms
                validation = validate_assistant_response(final_generation.raw_response)

            if not validation.valid_json:
                final_failure = "; ".join(validation.errors) or "Invalid structured response"

            system = capture_system_snapshot()
            result = self._to_result(
                case=case,
                generation=final_generation,
                total_latency_ms=total_latency_ms,
                retry_count=retry_count,
                valid_json=validation.valid_json,
                parsed_response=validation.parsed,
                validation_errors=validation.errors,
                final_failure=final_failure,
                cpu_percent=system.cpu_percent,
                memory_mb=system.memory_mb,
                memory_percent=system.memory_percent,
            )
            record_benchmark_result(result)
            span.set_attribute("response.valid_json", result.valid_json)
            span.set_attribute("response.retry_count", result.retry_count)
            span.set_attribute("latency.total_ms", result.total_latency_ms)
            logger.info(
                "benchmark_case_completed",
                model=result.model,
                temperature=result.temperature,
                prompt_id=result.prompt_id,
                valid_json=result.valid_json,
                retry_count=result.retry_count,
                total_latency_ms=result.total_latency_ms,
            )
            return result

    def _build_cases(
        self,
        prompts: list[BenchmarkPrompt],
        runs_per_prompt: int,
        selected_models: list[str] | None = None,
    ) -> list[BenchmarkCase]:
        configured_models = [model.name for model in self.models_config.models]
        models = selected_models or configured_models
        unknown_models = sorted(set(models) - set(configured_models))
        if unknown_models:
            console.print(f"[yellow]Warning: models not present in config/models.yaml: {', '.join(unknown_models)}[/yellow]")
        return [
            BenchmarkCase(prompt=prompt, model=model, temperature=temperature, repeat_index=repeat_index)
            for model in models
            for temperature in self.benchmark_config.temperatures
            for prompt in prompts
            for repeat_index in range(1, runs_per_prompt + 1)
        ]

    @staticmethod
    def _error_result(
        case: BenchmarkCase,
        error: Exception,
        retry_count: int = 0,
        elapsed_ms: float | None = None,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            prompt_id=case.prompt.id,
            prompt_category=case.prompt.category,
            model=case.model,
            temperature=case.temperature,
            repeat_index=case.repeat_index,
            raw_response="",
            parsed_response=None,
            reference_answer=case.prompt.reference_answer,
            valid_json=False,
            validation_errors=[str(error)],
            retry_count=retry_count,
            final_failure=str(error),
            time_to_first_token_ms=None,
            total_latency_ms=elapsed_ms if elapsed_ms is not None else getattr(error, "elapsed_ms", 0.0),
            output_tokens=0,
            tokens_per_second=0,
        )

    @staticmethod
    def _to_result(
        case: BenchmarkCase,
        generation: GenerationResponse,
        total_latency_ms: float,
        retry_count: int,
        valid_json: bool,
        parsed_response: dict | None,
        validation_errors: list[str],
        final_failure: str | None,
        cpu_percent: float | None,
        memory_mb: float | None,
        memory_percent: float | None,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            prompt_id=case.prompt.id,
            prompt_category=case.prompt.category,
            model=case.model,
            temperature=case.temperature,
            repeat_index=case.repeat_index,
            raw_response=generation.raw_response,
            parsed_response=parsed_response,
            reference_answer=case.prompt.reference_answer,
            valid_json=valid_json,
            validation_errors=validation_errors,
            retry_count=retry_count,
            final_failure=final_failure,
            time_to_first_token_ms=generation.timing.time_to_first_token_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=generation.timing.output_tokens,
            tokens_per_second=generation.timing.tokens_per_second,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
        )

    @staticmethod
    def _print_summary(output_path: Path, case_count: int) -> None:
        console.print(f"[green]Completed {case_count} benchmark cases.[/green]")
        console.print(f"Results: {output_path}")

