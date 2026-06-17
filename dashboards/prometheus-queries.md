# Prometheus Query Links

Prometheus is a query explorer, not a dashboard. Open a link below, then click **Execute** if the graph does not run automatically.

## Target Health

- [Scrape target status](http://localhost:9090/targets)
- [Configured metric names](http://localhost:9090/graph?g0.expr=%7B__name__%3D~%22slm_.*%22%7D&g0.tab=1)

## Benchmark Metrics

- [Average latency](http://localhost:9090/graph?g0.expr=slm_average_latency_ms&g0.tab=0)
- [P95 latency](http://localhost:9090/graph?g0.expr=slm_p95_latency_ms&g0.tab=0)
- [Average tokens per second](http://localhost:9090/graph?g0.expr=slm_average_tokens_per_second&g0.tab=0)
- [Average memory MB](http://localhost:9090/graph?g0.expr=slm_average_memory_mb&g0.tab=0)
- [JSON validation success rate](http://localhost:9090/graph?g0.expr=slm_json_validation_success_rate&g0.tab=0)
- [Retry rate](http://localhost:9090/graph?g0.expr=slm_retry_rate&g0.tab=0)
- [Quality score](http://localhost:9090/graph?g0.expr=slm_quality_score&g0.tab=0)
- [Output variance score](http://localhost:9090/graph?g0.expr=slm_output_variance_score&g0.tab=0)

## Recording Rules

After Prometheus reloads the rules, these shorter names are also available:

- `slm:average_latency_ms`
- `slm:p95_latency_ms`
- `slm:average_tokens_per_second`
- `slm:average_memory_mb`
- `slm:json_validation_success_rate`
- `slm:retry_rate`
- `slm:quality_score`
- `slm:output_variance_score`

