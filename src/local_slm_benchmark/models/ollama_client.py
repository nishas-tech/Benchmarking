"""Ollama HTTP client with streaming timing metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

import httpx

from local_slm_benchmark.models.schemas import GenerateRequest, GenerationResponse, GenerationTiming, schema_instructions


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout_seconds: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, request: GenerateRequest) -> GenerationResponse:
        prompt = self._build_prompt(request)
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []

        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": request.temperature},
        }
        if request.format:
            payload["format"] = request.format
        if request.system_prompt:
            payload["system"] = request.system_prompt

        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream("POST", f"{self.host}/api/generate", json=payload) as response:
                response.raise_for_status()
                for token in self._stream_response(response.iter_lines()):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunks.append(token)

        finished = time.perf_counter()
        raw_response = "".join(chunks)
        total_latency_ms = (finished - started) * 1000
        output_tokens = self._estimate_tokens(raw_response)
        tokens_per_second = output_tokens / max(finished - started, 0.001)

        timing = GenerationTiming(
            time_to_first_token_ms=((first_token_at - started) * 1000) if first_token_at else None,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
        )
        return GenerationResponse(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            raw_response=raw_response,
            timing=timing,
        )

    def _build_prompt(self, request: GenerateRequest) -> str:
        return f"{request.prompt.strip()}\n\n{schema_instructions()}"

    @staticmethod
    def _stream_response(lines: Iterator[str]) -> Iterator[str]:
        for line in lines:
            if not line:
                continue
            data = json.loads(line)
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Good enough for relative local benchmarks when tokenizer details vary by model.
        return max(1, len(text.split())) if text else 0

