# Local SLM Benchmark Report

Generated at: `2026-04-25T20:26:27.083073+00:00`

Results source: `results\benchmark-20260425T202332Z.jsonl`

## Executive Summary

This report compares local SLM benchmark runs across latency, throughput, structured JSON reliability, retry behavior, memory usage, output variance, and quality scoring.

## Methodology

The benchmark runner executes a configurable matrix of models, temperatures, prompts, and repeat counts. Each generation is validated against the required JSON schema, retried when invalid, persisted to JSONL, analyzed with pandas, compared with DeepDiff, and scored with the local quality scorer.

## Aggregate Results

| model | temperature | average_latency_ms | p95_latency_ms | average_tokens_per_second | average_memory_mb | json_validation_success_rate | retry_rate | quality_score | output_variance_score | run_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.2:3b | 0.000 | 5119.939 | 5542.626 | 2.703 | 61.941 | 1.000 | 0.000 | 0.538 | 0.000 | 2 |
| llama3.2:3b | 0.700 | 5972.227 | 7395.226 | 3.927 | 63.244 | 1.000 | 0.000 | 0.731 | 0.000 | 2 |

## Charts

![average_latency](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/average-latency-by-model.png)
![tokens_per_second](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/tokens-per-second-by-model.png)
![json_success](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/json-success-rate-by-model.png)
![quality_vs_latency](E:/AI-Workstuff/Projects/Benchmarking/reports/charts/quality-vs-latency.png)

## Interpretation Checklist

- Prefer models with low p95 latency and high tokens per second for interactive use.
- Prefer models with high JSON validation success and low retry rate for structured workflows.
- Compare quality score against latency before choosing the final local assistant model.
- Use variance score to understand how temperature changes determinism.

## Limitations

- Token counts are approximated from whitespace-separated output unless model tokenizer integration is added later.
- Quality scoring is local and lightweight by default; DeepEval can be expanded with a stronger judge model later.
- Hardware, Ollama version, and model quantization can significantly change benchmark results.
