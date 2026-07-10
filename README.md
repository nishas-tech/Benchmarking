# Local SLM Benchmarking Assistant

This project benchmarks local Small Language Models through Ollama and evaluates them across latency, throughput, structured JSON reliability, determinism, quality, and hardware usage.

See `docs/REQUIREMENTS.md` for the project reference architecture and implementation goals.

## Two-Stage Workflow

This project is designed for a deliberate two-stage workflow:

1. **Benchmark machine**: run the expensive generation matrix and save portable result files.
2. **Analysis machine**: copy those files over and generate reports, quality scores, and optional Grafana metrics.

That split keeps long-running model comparisons off your daily development machine and makes the benchmark reproducible on stronger hardware.

### Stage 1: Benchmark Machine

```bash
python -m venv .venv
python -m pip install -e .
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull phi4
slm-benchmark benchmark run --runs-per-prompt 3
```

This writes two files per run:

- `results/benchmark-YYYYMMDDTHHMMSSZ.jsonl`
- `results/benchmark-YYYYMMDDTHHMMSSZ.meta.json`

Copy both files to the analysis machine. The metadata file preserves benchmark-machine hardware, Ollama version, model list, prompt counts, and total benchmark duration.

### Stage 2: Analysis Machine

```bash
python -m venv .venv
python -m pip install -e .
ollama pull llama3.2:3b
slm-benchmark report generate --results results/benchmark-YYYYMMDDTHHMMSSZ.jsonl
```

For faster iteration without judge-model calls:

```bash
slm-benchmark report generate --quality-scorer lexical
```

Optional post-run metrics export for Grafana:

```bash
slm-benchmark metrics export --results results/benchmark-YYYYMMDDTHHMMSSZ.jsonl
```

## Setup

```bash
python -m venv .venv
python -m pip install -e .
ollama pull llama3.2:3b
slm-benchmark env
slm-benchmark prompts validate
```

## Smoke Test

```bash
slm-benchmark generate --model llama3.2:3b --prompt "Return JSON with a one sentence summary."
slm-benchmark benchmark run --model llama3.2:3b --limit 2 --runs-per-prompt 1
slm-benchmark report generate
```

## Benchmark Matrix

The full benchmark uses all prompts in `prompts/benchmark_prompts.json`, all configured models, and all configured temperatures. With the default config, that is approximately:

```text
3 models x 2 temperatures x 55 prompts x 3 runs = 990 generations
```

To benchmark only installed models:

```bash
slm-benchmark benchmark run --model llama3.2:3b --model mistral:7b --runs-per-prompt 3
```

## DeepEval Quality Scoring

Quality scoring is configured in `config/benchmark.yaml`:

```yaml
quality_scorer: deepeval
judge_model: llama3.2:3b
judge_temperature: 0.0
```

By default, report generation uses **DeepEval with a local Ollama judge**. The judge is separate from the models being benchmarked. This keeps the evaluation local/private and avoids cloud API costs.

DeepEval produces:

- an overall quality score
- separate dimension scores for correctness, relevance, completeness, instruction following, and conciseness

Environment overrides:

```bash
SLM_QUALITY_SCORER=deepeval
SLM_JUDGE_MODEL=llama3.2:3b
SLM_JUDGE_TEMPERATURE=0.0
```

Use `--quality-scorer lexical` when you want a fast deterministic score without judge-model calls.

## Metrics And Dashboards

Grafana is optional and intended for **offline/post-run analysis**, not live production monitoring.

```bash
slm-benchmark serve --host 0.0.0.0 --port 8000
docker compose -f docker-compose.observability.yml up
slm-benchmark metrics export --results results/benchmark-YYYYMMDDTHHMMSSZ.jsonl
```

Open:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Dashboard: `Local SLM Benchmarking`

The analysis machine does not need to rerun the benchmark. `metrics export` reads the saved JSONL file, computes analysis metrics, and exposes them through the metrics bridge consumed by Grafana.

## Main Commands

- `slm-benchmark env`: print local environment information.
- `slm-benchmark prompts validate`: validate the benchmark prompt file.
- `slm-benchmark generate`: run one prompt against one Ollama model.
- `slm-benchmark benchmark run`: execute the benchmark matrix on the benchmark machine.
- `slm-benchmark report generate`: generate a Markdown report from saved results on the analysis machine.
- `slm-benchmark serve`: start the FastAPI app and Prometheus `/metrics` endpoint.
- `slm-benchmark metrics export`: backfill Prometheus/Grafana metrics from a saved JSONL result file.

## Report Output

Reports include:

- hardware/environment metadata from the benchmark machine
- model and prompt context
- latency, throughput, memory, reliability, and quality sections
- temperature/determinism comparison
- separate DeepEval quality dimensions
- recommended model configurations for interactive, structured, and balanced use cases
- charts for latency, p95 latency, TTFT, throughput, JSON success, retry rate, memory, and quality vs latency
