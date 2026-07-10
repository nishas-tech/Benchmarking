"""Ollama HTTP client with streaming timing metrics."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from local_slm_benchmark.models.schemas import GenerateRequest, GenerationResponse, GenerationTiming, schema_instructions


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama rejects or fails a generation request."""

    def __init__(self, message: str, status_code: int | None = None, elapsed_ms: float = 0.0) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.elapsed_ms = elapsed_ms


@dataclass(frozen=True)
class StreamChunk:
    text: str
    eval_count: int | None = None


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout_seconds: float = 300.0) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate(self, request: GenerateRequest) -> GenerationResponse:
        prompt = self._build_prompt(request)
        started = time.perf_counter()
        first_token_at: float | None = None
        chunks: list[str] = []
        eval_count: int | None = None

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

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with client.stream("POST", f"{self.host}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    for chunk in self._stream_response(response.iter_lines()):
                        if first_token_at is None and chunk.text:
                            first_token_at = time.perf_counter()
                        if chunk.text:
                            chunks.append(chunk.text)
                        if chunk.eval_count is not None:
                            eval_count = chunk.eval_count
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            detail = self._response_detail(exc.response)
            raise OllamaGenerationError(
                f"Ollama returned HTTP {exc.response.status_code} for model '{request.model}': {detail}",
                status_code=exc.response.status_code,
                elapsed_ms=elapsed_ms,
            ) from exc
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            raise OllamaGenerationError(
                f"Ollama request failed for model '{request.model}': {exc}",
                elapsed_ms=elapsed_ms,
            ) from exc

        finished = time.perf_counter()
        raw_response = "".join(chunks)
        total_latency_ms = (finished - started) * 1000
        estimated_output_tokens = self._estimate_tokens(raw_response)
        output_tokens = eval_count if eval_count is not None else estimated_output_tokens
        tokens_per_second = output_tokens / max(finished - started, 0.001)

        timing = GenerationTiming(
            time_to_first_token_ms=((first_token_at - started) * 1000) if first_token_at else None,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            estimated_output_tokens=estimated_output_tokens,
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
    def _stream_response(lines: Iterator[str]) -> Iterator[StreamChunk]:
        for line in lines:
            if not line:
                continue
            data = json.loads(line)
            eval_count = data.get("eval_count")
            yield StreamChunk(
                text=data.get("response", ""),
                eval_count=int(eval_count) if eval_count is not None else None,
            )
            if data.get("done"):
                break

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.split())) if text else 0

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            response.read()
            detail = response.text.strip()
        except Exception:
            detail = ""
        return detail or response.reason_phrase or "unknown error"
