"""Benchmark matrix runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from local_slm_benchmark.benchmark.metadata import (
    capture_run_metadata,
    finalize_run_metadata,
    write_run_metadata,
)
from local_slm_benchmark.benchmark.results import append_result, new_results_path
from local_slm_benchmark.benchmark.system import capture_system_snapshot
from local_slm_benchmark.config import (
    BenchmarkConfig,
    ModelsConfig,
    load_benchmark_config,
    load_models_config,
    model_quantization,
)
from local_slm_benchmark.models.ollama_client import OllamaClient, OllamaGenerationError
from local_slm_benchmark.models.schemas import BenchmarkPrompt, BenchmarkResult, GenerateRequest, GenerationResponse
from local_slm_benchmark.observability.logging import get_logger
from local_slm_benchmark.observability.metrics import record_benchmark_result
from local_slm_benchmark.observability.persistent_metrics import record_persisted_benchmark_result
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
        selected_runs = runs_per_prompt or self.benchmark_config.runs_per_prompt
        configured_models = [model.name for model in self.models_config.models]
        selected_models = models or configured_models
        cases = self._build_cases(prompts, selected_runs, selected_models)
        output_path = new_results_path(self.benchmark_config.results_dir)
        started_at = datetime.now(UTC)

        metadata = capture_run_metadata(
            results_path=output_path,
            models_config=self.models_config,
            benchmark_config=self.benchmark_config,
            prompts=prompts,
            case_count=len(cases),
            runs_per_prompt=selected_runs,
            selected_models=selected_models,
        )
        write_run_metadata(metadata, output_path)
        logger.info(
            "benchmark_started",
            results_path=str(output_path),
            case_count=len(cases),
            models=selected_models,
        )

        with Progress() as progress:
            task = progress.add_task("Running benchmark", total=len(cases))
            for case in cases:
                result = self.run_case(case)
                append_result(output_path, result)
                progress.update(task, advance=1)

        finalize_run_metadata(output_path, started_at=started_at, finished_at=datetime.now(UTC))
        self._print_summary(output_path, len(cases))
        logger.info("benchmark_completed", results_path=str(output_path), case_count=len(cases))
        return output_path

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        with tracer.start_as_current_span("benchmark.run_case") as span:
            span.set_attribute("model.name", case.model)
            span.set_attribute("model.temperature", case.temperature)
            span.set_attribute("prompt.id", case.prompt.id)
            span.set_attribute("prompt.category", case.prompt.category)
            request = GenerateRequest(model=case.model, prompt=case.prompt.prompt, temperature=case.temperature)
            generation_latency_ms = 0.0
            retry_latency_ms = 0.0
            retry_count = 0
            final_failure: str | None = None
            final_generation: GenerationResponse | None = None
            validation = validate_assistant_response("")

            try:
                final_generation = self.client.generate(request)
                generation_latency_ms = final_generation.timing.total_latency_ms
                validation = validate_assistant_response(final_generation.raw_response)
            except OllamaGenerationError as exc:
                result = self._error_result(case, exc)
                record_benchmark_result(result)
                record_persisted_benchmark_result(result)
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

            while not validation.valid_json and retry_count < self.benchmark_config.max_retries:
                retry_count += 1
                logger.info(
                    "benchmark_retry_started",
                    model=case.model,
                    prompt_id=case.prompt.id,
                    retry_count=retry_count,
                    error_category=validation.error_category,
                )
                retry_prompt = build_retry_prompt(
                    case.prompt.prompt,
                    final_generation.raw_response,
                    validation.errors,
                )
                try:
                    retry_generation = self.client.generate(
                        GenerateRequest(model=case.model, prompt=retry_prompt, temperature=case.temperature)
                    )
                except OllamaGenerationError as exc:
                    result = self._error_result(
                        case,
                        exc,
                        retry_count=retry_count,
                        elapsed_ms=generation_latency_ms + retry_latency_ms + exc.elapsed_ms,
                        generation_latency_ms=generation_latency_ms,
                        retry_latency_ms=retry_latency_ms,
                        validation=validation,
                    )
                    record_benchmark_result(result)
                    record_persisted_benchmark_result(result)
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
                retry_latency_ms += retry_generation.timing.total_latency_ms
                final_generation = retry_generation
                validation = validate_assistant_response(final_generation.raw_response)
                if validation.valid_json:
                    logger.info(
                        "benchmark_retry_succeeded",
                        model=case.model,
                        prompt_id=case.prompt.id,
                        retry_count=retry_count,
                    )
                else:
                    logger.warning(
                        "benchmark_validation_failed",
                        model=case.model,
                        prompt_id=case.prompt.id,
                        retry_count=retry_count,
                        error_category=validation.error_category,
                    )

            if not validation.valid_json:
                final_failure = "; ".join(validation.errors) or "Invalid structured response"

            assert final_generation is not None
            system = capture_system_snapshot()
            result = self._to_result(
                case=case,
                generation=final_generation,
                generation_latency_ms=generation_latency_ms,
                retry_latency_ms=retry_latency_ms,
                retry_count=retry_count,
                validation=validation,
                final_failure=final_failure,
                system=system,
            )
            record_benchmark_result(result)
            record_persisted_benchmark_result(result)
            span.set_attribute("response.valid_json", result.valid_json)
            span.set_attribute("response.retry_count", result.retry_count)
            span.set_attribute("response.output_tokens", result.output_tokens)
            span.set_attribute("latency.time_to_first_token_ms", result.time_to_first_token_ms or 0)
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

    def _error_result(
        self,
        case: BenchmarkCase,
        error: Exception,
        retry_count: int = 0,
        elapsed_ms: float | None = None,
        generation_latency_ms: float | None = None,
        retry_latency_ms: float = 0.0,
        validation=None,
    ) -> BenchmarkResult:
        system = capture_system_snapshot()
        total_latency_ms = elapsed_ms if elapsed_ms is not None else getattr(error, "elapsed_ms", 0.0)
        validation = validation or validate_assistant_response("")
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
            valid_json_parse=validation.valid_json_parse,
            valid_schema=validation.valid_schema,
            schema_error_category=validation.error_category or "generation_error",
            validation_errors=[str(error)],
            retry_count=retry_count,
            retry_succeeded=False,
            final_failure=str(error),
            time_to_first_token_ms=None,
            generation_latency_ms=generation_latency_ms,
            retry_latency_ms=retry_latency_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=0,
            estimated_output_tokens=0,
            tokens_per_second=0,
            model_quantization=model_quantization(self.models_config, case.model),
            cpu_percent=system.cpu_percent,
            memory_mb=system.memory_mb,
            memory_percent=system.memory_percent,
            gpu_name=system.gpu_name,
            gpu_memory_used_mb=system.gpu_memory_used_mb,
            gpu_utilization_percent=system.gpu_utilization_percent,
        )

    def _to_result(
        self,
        case: BenchmarkCase,
        generation: GenerationResponse,
        generation_latency_ms: float,
        retry_latency_ms: float,
        retry_count: int,
        validation,
        final_failure: str | None,
        system,
    ) -> BenchmarkResult:
        total_latency_ms = generation_latency_ms + retry_latency_ms
        tokens_per_second = generation.timing.output_tokens / max(total_latency_ms / 1000, 0.001)
        return BenchmarkResult(
            prompt_id=case.prompt.id,
            prompt_category=case.prompt.category,
            model=case.model,
            temperature=case.temperature,
            repeat_index=case.repeat_index,
            raw_response=generation.raw_response,
            parsed_response=validation.parsed,
            reference_answer=case.prompt.reference_answer,
            valid_json=validation.valid_json,
            valid_json_parse=validation.valid_json_parse,
            valid_schema=validation.valid_schema,
            schema_error_category=validation.error_category,
            validation_errors=validation.errors,
            retry_count=retry_count,
            retry_succeeded=retry_count > 0 and validation.valid_json,
            final_failure=final_failure,
            time_to_first_token_ms=generation.timing.time_to_first_token_ms,
            generation_latency_ms=generation_latency_ms,
            retry_latency_ms=retry_latency_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=generation.timing.output_tokens,
            estimated_output_tokens=generation.timing.estimated_output_tokens,
            tokens_per_second=tokens_per_second,
            model_quantization=model_quantization(self.models_config, case.model),
            cpu_percent=system.cpu_percent,
            memory_mb=system.memory_mb,
            memory_percent=system.memory_percent,
            gpu_name=system.gpu_name,
            gpu_memory_used_mb=system.gpu_memory_used_mb,
            gpu_utilization_percent=system.gpu_utilization_percent,
        )

    @staticmethod
    def _print_summary(output_path: Path, case_count: int) -> None:
        console.print(f"[green]Completed {case_count} benchmark cases.[/green]")
        console.print(f"Results: {output_path}")
        console.print(f"Metadata: {output_path.with_suffix('.meta.json')}")
