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

## Main Commands

- `slm-benchmark env`: print local environment information.
- `slm-benchmark prompts validate`: validate the benchmark prompt file.
- `slm-benchmark generate`: run one prompt against one Ollama model.
- `slm-benchmark benchmark run`: execute the benchmark matrix.
- `slm-benchmark report generate`: generate a Markdown report from saved results.

