# Local SLM Benchmarking Assistant

This project benchmarks local Small Language Models through Ollama and evaluates them across latency, throughput, structured JSON reliability, determinism, quality, and hardware usage.

See `REQUIREMENTS.md` for the project reference architecture and implementation goals.

## Setup

```bash
python -m venv .venv
python -m pip install -e .
ollama pull llama3.2:3b
slm-benchmark env
slm-benchmark prompts validate
```

## Smoke Test

Use the smoke test to verify the local project, Ollama connection, JSON validation, result persistence, and report generation with a small run.

```bash
slm-benchmark generate --model llama3.2:3b --prompt "Return JSON with a one sentence summary."
slm-benchmark benchmark run --model llama3.2:3b --limit 2 --runs-per-prompt 1
slm-benchmark report generate
```

The smoke benchmark filters to `llama3.2:3b` because setup only pulls that model. It runs 2 prompts across the configured temperatures and writes a small JSONL result file under `results/`.

## Benchmark Run

For the full comparison, pull every model listed in `config/models.yaml` first:

```bash
ollama pull llama3.2:3b
ollama pull mistral:7b
ollama pull phi4
```

Then run the full configured benchmark matrix:

```bash
slm-benchmark benchmark run --runs-per-prompt 3
slm-benchmark report generate
```

The full benchmark uses all 50 prompts in `prompts/benchmark_prompts.json`, all configured models, and all configured temperatures. With the default config, that is:

```text
3 models x 2 temperatures x 50 prompts x 3 runs = 900 generations
```

To benchmark only the models installed on your machine, pass one or more `--model` options:

```bash
slm-benchmark benchmark run --model llama3.2:3b --model mistral:7b --runs-per-prompt 3
```

## Metrics And Dashboards

Start the FastAPI metrics endpoint in one terminal:

```bash
slm-benchmark serve --host 0.0.0.0 --port 8000
```

Start Prometheus and Grafana in another terminal:

```bash
docker compose -f docker-compose.observability.yml up
```

Open the dashboards:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Grafana login: `admin` / `admin`
- Dashboard: `Local SLM Benchmarking`

Prometheus scrapes FastAPI at `http://host.docker.internal:8000/metrics`. The benchmark CLI also writes persisted metric snapshots to `results/prometheus-metrics.json`, and the FastAPI `/metrics` endpoint exposes those snapshots so Grafana can show CLI-produced runtime metrics and post-run analysis metrics.

Prometheus does not auto-render a dashboard on the query page. Use Grafana for the visual dashboard, or open the ready-made Prometheus query links in `dashboards/prometheus-queries.md`.

Recommended dashboard flow:

```bash
slm-benchmark serve --host 0.0.0.0 --port 8000
docker compose -f docker-compose.observability.yml up
slm-benchmark benchmark run --model llama3.2:3b --limit 2 --runs-per-prompt 1
slm-benchmark report generate
```

If you already have a benchmark JSONL file and want to populate Grafana without rerunning the benchmark, export the saved results into the metrics bridge:

```bash
slm-benchmark metrics export --results results/benchmark-YYYYMMDDTHHMMSSZ.jsonl
```

Verify the metrics endpoint:

```bash
curl http://localhost:8000/metrics/status
curl http://localhost:8000/metrics
```

In Prometheus, check `Status > Targets`. The `local-slm-benchmark` target should be `UP`. If it is down, make sure `slm-benchmark serve --host 0.0.0.0 --port 8000` is still running.

## Main Commands

- `slm-benchmark env`: print local environment information.
- `slm-benchmark prompts validate`: validate the benchmark prompt file.
- `slm-benchmark generate`: run one prompt against one Ollama model.
- `slm-benchmark benchmark run`: execute the benchmark matrix.
- `slm-benchmark report generate`: generate a Markdown report from saved results.
- `slm-benchmark serve`: start the FastAPI app and Prometheus `/metrics` endpoint.
- `slm-benchmark metrics export`: backfill Prometheus/Grafana metrics from a saved JSONL result file.

