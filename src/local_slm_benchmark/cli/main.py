"""Typer CLI for local SLM benchmarking."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from local_slm_benchmark.config import load_benchmark_config, load_models_config, project_path
from local_slm_benchmark.models.ollama_client import OllamaClient
from local_slm_benchmark.models.schemas import GenerateRequest
from local_slm_benchmark.prompts import load_prompts
from local_slm_benchmark.validation.parser import validate_assistant_response


console = Console()
app = typer.Typer(help="Local SLM benchmarking assistant.")
prompts_app = typer.Typer(help="Prompt file utilities.")
benchmark_app = typer.Typer(help="Benchmark commands.")
report_app = typer.Typer(help="Report commands.")
metrics_app = typer.Typer(help="Metrics export commands.")
app.add_typer(prompts_app, name="prompts")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(report_app, name="report")
app.add_typer(metrics_app, name="metrics")

@app.command()
def help(ctx: typer.Context) -> None:
    """Show help for all commands and subcommands."""
    typer.echo(ctx.parent.get_help())  # type: ignore[union-attr]

@app.command()
def env() -> None:
    """Print local environment information."""

    models_config = load_models_config()
    benchmark_config = load_benchmark_config()
    table = Table(title="Local SLM Benchmark Environment")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Ollama host", models_config.ollama_host)
    table.add_row("Configured models", ", ".join(model.name for model in models_config.models))
    table.add_row("Prompt file", benchmark_config.prompts_path)
    table.add_row("Results dir", str(project_path(benchmark_config.results_dir)))
    table.add_row("Quality scorer", benchmark_config.quality_scorer)
    table.add_row("Judge model", benchmark_config.judge_model or "(first configured model)")
    console.print(table)


@app.command()
def generate(
    prompt: Annotated[str, typer.Option(help="Prompt to send to Ollama.")],
    model: Annotated[str | None, typer.Option(help="Ollama model name.")] = None,
    temperature: Annotated[float, typer.Option(help="Generation temperature.")] = 0.0,
) -> None:
    """Run one prompt against one local Ollama model."""

    models_config = load_models_config()
    selected_model = model or (models_config.models[0].name if models_config.models else "llama3.2:3b")
    client = OllamaClient(models_config.ollama_host)
    response = client.generate(GenerateRequest(model=selected_model, prompt=prompt, temperature=temperature))
    validation = validate_assistant_response(response.raw_response)

    console.print(response.raw_response)
    console.print(
        f"[bold]Latency:[/bold] {response.timing.total_latency_ms:.1f} ms | "
        f"[bold]TTFT:[/bold] {response.timing.time_to_first_token_ms or 0:.1f} ms | "
        f"[bold]Tokens/sec:[/bold] {response.timing.tokens_per_second:.2f} | "
        f"[bold]Valid JSON:[/bold] {validation.valid_json}"
    )
    if validation.errors:
        console.print("[red]Validation errors:[/red]")
        for error in validation.errors:
            console.print(f"- {error}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="FastAPI host.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="FastAPI port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable Uvicorn reload.")] = False,
) -> None:
    """Start the FastAPI app that exposes /generate and /metrics."""

    import uvicorn

    uvicorn.run("local_slm_benchmark.api.app:app", host=host, port=port, reload=reload)


@prompts_app.command("validate")
def validate_prompts(
    path: Annotated[Path | None, typer.Option(help="Prompt JSON file path.")] = None,
) -> None:
    """Validate benchmark prompt file structure."""

    benchmark_config = load_benchmark_config()
    prompt_path = path or Path(benchmark_config.prompts_path)
    prompts = load_prompts(prompt_path)
    console.print(f"[green]Validated {len(prompts)} prompts from {prompt_path}[/green]")


@benchmark_app.command("run")
def benchmark_run(
    limit: Annotated[int | None, typer.Option(help="Limit prompts for a smoke run.")] = None,
    runs_per_prompt: Annotated[int | None, typer.Option(help="Override repeat count.")] = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="Only benchmark this model. Can be provided more than once."),
    ] = None,
) -> None:
    """Run the benchmark matrix."""

    from local_slm_benchmark.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner()
    output_path = runner.run(limit=limit, runs_per_prompt=runs_per_prompt, models=model)
    console.print(f"[green]Benchmark results written to {output_path}[/green]")


@report_app.command("generate")
def report_generate(
    results: Annotated[Path | None, typer.Option(help="Results JSONL file.")] = None,
    quality_scorer: Annotated[
        str | None,
        typer.Option(help="Quality scorer to use: deepeval or lexical."),
    ] = None,
) -> None:
    """Generate a Markdown report from saved benchmark results."""

    from local_slm_benchmark.reporting.report import generate_report

    report_path = generate_report(results_path=results, quality_scorer=quality_scorer)
    console.print(f"[green]Report written to {report_path}[/green]")


@metrics_app.command("export")
def metrics_export(
    results: Annotated[Path | None, typer.Option(help="Results JSONL file. Defaults to latest result file.")] = None,
    quality_scorer: Annotated[
        str | None,
        typer.Option(help="Quality scorer to use: deepeval or lexical."),
    ] = None,
) -> None:
    """Export saved benchmark results into the Prometheus metrics bridge."""

    from local_slm_benchmark.benchmark.results import latest_results_file
    from local_slm_benchmark.config import load_benchmark_config
    from local_slm_benchmark.observability.persistent_metrics import METRICS_PATH, persist_analysis_summary
    from local_slm_benchmark.reporting.analysis import analyze_results

    config = load_benchmark_config()
    selected_results = results or latest_results_file(config.results_dir)
    if selected_results is None:
        raise typer.BadParameter("No benchmark result files found.")

    analysis = analyze_results(selected_results, quality_scorer=quality_scorer)
    persist_analysis_summary(analysis.summary)
    console.print(f"[green]Exported analysis metrics from {selected_results}[/green]")
    console.print(f"Metrics bridge file: {METRICS_PATH}")


if __name__ == "__main__":
    app()

