# Local SLM Benchmark Report

Generated at: `2026-07-18T19:48:16.101913+00:00`

Results source: `results\benchmark-20260712T210827Z.jsonl`

## Executive Summary

This report compares local SLM benchmark runs across latency, throughput, structured JSON reliability, retry behavior, memory usage, output variance, temperature sensitivity, and quality scoring. Use the recommendation section for a practical model choice based on interactive speed, structured reliability, or balanced usage.

## Hardware And Environment

- Platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`
- Python: `3.14.0`
- CPU count: `16`
- Total memory: `131072 MB`
- Ollama host: `http://localhost:11434`
- Ollama version: `0.30.10`
- Benchmark duration: `6424585 ms`

## Model List And Configuration

| Model | Label | Quantization |
| --- | --- | --- |
| phi4 | Phi 4 | default |

Temperatures: `0, 0.7`
Runs per prompt: `3`

## Prompt Set Description

- Prompt count: `55`
- Categories:
- `classification`: 5
- `coding`: 5
- `comparison`: 5
- `extraction`: 5
- `observability`: 5
- `planning`: 5
- `reasoning`: 5
- `reporting`: 5
- `safety`: 5
- `summarization`: 5
- `transformation`: 5

## Benchmark Methodology

The benchmark runner executes a configurable matrix of models, temperatures, prompts, and repeat counts on the benchmark machine. Each generation is validated against the required JSON schema, retried when invalid, persisted to JSONL, and paired with a metadata file describing the benchmark environment. Analysis on a separate machine loads the saved JSONL file, runs DeepDiff variance analysis, `deepeval` quality evaluation, and pandas aggregation to produce this report.

## Latency Comparison

Key latency metrics are summarized below. Lower average and p95 latency generally indicate a better interactive experience.

| model | temperature | average_latency_ms | average_generation_latency_ms | average_retry_latency_ms | p95_latency_ms | average_time_to_first_token_ms | average_tokens_per_second | average_memory_mb | average_gpu_memory_used_mb | json_validation_success_rate | json_parse_success_rate | schema_validation_success_rate | retry_rate | retry_success_rate | final_failure_rate | quality_score | output_variance_score | run_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4 | 0.000 | 2501.028 | 2501.028 | 0.000 | 6695.385 | 398.910 | 52.043 | 128.389 |  | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.559 | 0.000 | 165 |
| gemma4 | 0.700 | 2559.641 | 2357.914 | 201.727 | 7009.551 | 382.185 | 53.441 | 132.023 |  | 0.994 | 0.994 | 0.994 | 0.036 | 0.030 | 0.006 | 0.559 | 1.358 | 165 |
| phi4 | 0.000 | 3944.888 | 3944.888 | 0.000 | 8516.108 | 237.718 | 28.790 | 134.836 |  | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.557 | 0.073 | 165 |
| phi4 | 0.700 | 3758.533 | 3728.905 | 29.628 | 8868.540 | 139.431 | 32.143 | 137.534 |  | 0.994 | 0.994 | 0.994 | 0.006 | 0.000 | 0.006 | 0.550 | 1.921 | 165 |

## Throughput Comparison

Average tokens per second and time to first token are included in the aggregate table above and in the dedicated charts below.

## Memory Usage Comparison

Process memory and optional GPU memory usage were captured during benchmark runs on the benchmark machine.

## JSON Reliability Comparison

- JSON parse success rate: `99.7%`
- Schema validation success rate: `99.7%`
- Retry rate: `1.1%`
- Retry success rate: `71.4%`
- Final failure rate: `0.3%`
- Schema error categories: `{'none': 658, 'json_parse': 2}`

## Temperature And Determinism Analysis

This section compares repeat-run variance within each temperature and the distance between outputs at different temperatures for the same prompt.

| model | low_temperature | high_temperature | average_low_temp_variance | average_high_temp_variance | average_cross_temperature_distance | prompt_count |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4 | 0.000 | 0.700 | 0.000 | 1.358 | 2.000 | 55.000 |
| phi4 | 0.000 | 0.700 | 0.073 | 1.921 | 2.873 | 55.000 |

## Quality Evaluation Summary

Quality scoring uses DeepEval with a local Ollama judge when enabled. The dimension table below breaks correctness, relevance, completeness, instruction following, and conciseness into separate scores.

| model | temperature | correctness | relevance | completeness | instruction_following | conciseness |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4 | 0.000 | 0.481 | 0.481 | 0.481 | 1.000 | 0.352 |
| gemma4 | 0.700 | 0.470 | 0.470 | 0.470 | 1.000 | 0.384 |
| phi4 | 0.000 | 0.465 | 0.465 | 0.465 | 1.000 | 0.390 |
| phi4 | 0.700 | 0.456 | 0.456 | 0.456 | 1.000 | 0.383 |

## Recommended Model Configuration

- **Interactive**: `gemma4` at temperature `0` (score `0.99`). Low average latency (2501 ms) and strong throughput (52.0 tok/s) make this the best fit for responsive local use.
- **Structured**: `gemma4` at temperature `0` (score `0.99`). High JSON success (100.0%) and lower retry/failure pressure make this the safest choice for structured assistant workflows.
- **Balanced**: `gemma4` at temperature `0` (score `0.99`). Balanced latency (2501 ms), quality (0.56), and reliability (100.0%) provide the best overall tradeoff.

## Charts

![average_latency](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/average-latency-by-model.png)
![p95_latency](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/p95-latency-by-model.png)
![time_to_first_token](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/time-to-first-token-by-model.png)
![tokens_per_second](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/tokens-per-second-by-model.png)
![json_success](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/json-success-rate-by-model.png)
![retry_rate](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/retry-rate-by-model.png)
![memory_usage](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/memory-usage-by-model.png)
![quality_vs_latency](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/quality-vs-latency.png)

## Interpretation Checklist

- Prefer models with low p95 latency and high tokens per second for interactive use.
- Prefer models with high JSON validation success and low retry rate for structured workflows.
- Compare quality score against latency before choosing the final local assistant model.
- Use variance and temperature comparison tables to understand determinism tradeoffs.

## Limitations And Next Steps

- Token counts prefer Ollama `eval_count` when available and fall back to a whitespace estimate otherwise.
- Quality scoring uses DeepEval with a local Ollama judge by default; runs fall back to lexical overlap if DeepEval is unavailable or fails.
- DeepEval dimension scoring is slower than lexical overlap because each result may require multiple judge-model calls.
- GPU metrics are included when `nvidia-smi` is available on the benchmark machine.
- Hardware, Ollama version, and model quantization can significantly change benchmark results.
