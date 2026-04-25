# Local SLM Benchmarking Assistant

This project benchmarks local Small Language Models through Ollama and evaluates them across latency, throughput, structured JSON reliability, determinism, quality, and hardware usage.

See `REQUIREMENTS.md` for the project reference architecture and implementation goals.

## Quick Start

```bash
python -m venv .venv
python -m pip install -e .
ollama pull llama3.2:3b
slm-benchmark env
slm-benchmark generate --model llama3.2:3b --prompt "Return JSON with a one sentence summary."
slm-benchmark benchmark run --limit 2 --runs-per-prompt 1
```

## Main Commands

- `slm-benchmark env`: print local environment information.
- `slm-benchmark prompts validate`: validate the benchmark prompt file.
- `slm-benchmark generate`: run one prompt against one Ollama model.
- `slm-benchmark benchmark run`: execute the benchmark matrix.
- `slm-benchmark report generate`: generate a Markdown report from saved results.

