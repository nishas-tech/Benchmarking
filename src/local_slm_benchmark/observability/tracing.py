"""OpenTelemetry tracing helpers."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor


_CONFIGURED = False


def configure_tracing(service_name: str = "local-slm-benchmark") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def get_tracer(name: str = "local_slm_benchmark"):
    return trace.get_tracer(name)

