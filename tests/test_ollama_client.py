import httpx

from local_slm_benchmark.models.ollama_client import OllamaClient


def test_response_detail_handles_unread_streaming_response() -> None:
    response = httpx.Response(
        500,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        stream=httpx.ByteStream(b"model load failed"),
    )

    assert OllamaClient._response_detail(response) == "model load failed"


def test_stream_response_reads_eval_count_from_final_chunk() -> None:
    lines = iter(
        [
            '{"response":"hello","done":false}',
            '{"response":"","done":true,"eval_count":42}',
        ]
    )
    chunks = list(OllamaClient._stream_response(lines))
    assert chunks[0].text == "hello"
    assert chunks[-1].eval_count == 42

