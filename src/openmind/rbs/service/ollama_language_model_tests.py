import json
import logging

import pytest

from openmind.rbs.service.ollama_language_model import OllamaLanguageModel

pytestmark = pytest.mark.log_level("INFO")


def test_the_prompt_is_posted_without_streaming_or_thinking_and_the_thinking_is_removed() -> None:
    posted: list[tuple[str, dict[str, object], float]] = []

    def transport(url: str, body: bytes, timeout: float) -> bytes:
        posted.append((url, json.loads(body), timeout))
        return json.dumps({"response": "<think>counting</think>\n A knight fork wins a piece. "}).encode()

    model = OllamaLanguageModel("http://127.0.0.1:11434/", "qwen3:8b", 30.0, transport)

    assert (model.name, model.complete("Explain.")) == ("qwen3:8b", "A knight fork wins a piece.")
    assert posted == [
        (
            "http://127.0.0.1:11434/api/generate",
            {"model": "qwen3:8b", "prompt": "Explain.", "stream": False, "think": False, "options": {"temperature": 0}},
            30.0,
        )
    ]


def test_no_answer_or_an_unreadable_one_gives_none(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="openmind.rbs")

    def unreachable(url: str, body: bytes, timeout: float) -> bytes:
        raise OSError("connection refused")

    assert OllamaLanguageModel("http://nowhere:1", "qwen3:8b", 1.0, unreachable).complete("Explain.") is None
    assert OllamaLanguageModel("http://x:1", "qwen3:8b", 1.0, lambda url, body, timeout: b"not json").complete("E") is None
    assert OllamaLanguageModel("http://x:1", "qwen3:8b", 1.0, lambda url, body, timeout: b'{"response": ""}').complete("E") is None
    assert "qwen3:8b at http://nowhere:1 gave no answer: connection refused" in caplog.messages
