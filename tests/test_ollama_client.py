import httpx

from local_slm_benchmark.models.ollama_client import OllamaClient


def test_response_detail_handles_unread_streaming_response() -> None:
    response = httpx.Response(
        500,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        stream=httpx.ByteStream(b"model load failed"),
    )

    assert OllamaClient._response_detail(response) == "model load failed"

