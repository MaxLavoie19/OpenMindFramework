import json
import logging
import re
import urllib.request
from collections.abc import Callable

from openmind.rbs.constant.explanation_constant import DEFAULT_EXPLAINER_TIMEOUT

logger = logging.getLogger(__name__)

#: Posts a JSON body to a URL within a timeout and gives the response's body.
type Transport = Callable[[str, bytes, float], bytes]


def _post(url: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - the URL is the user's own server
        return response.read()


class OllamaLanguageModel:
    """A language model served by Ollama: posts the prompt to <url>/api/generate without streaming or thinking, at
    temperature 0, and gives the answer with any <think> section removed. The standard library is its only dependency.
    An unreachable server, a timeout or an answer it can't read gives None, logged as a warning."""

    def __init__(self, url: str, model: str, timeout: float = DEFAULT_EXPLAINER_TIMEOUT, transport: Transport = _post) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._transport = transport

    @property
    def name(self) -> str:
        return self._model

    def complete(self, prompt: str) -> str | None:
        body = json.dumps(
            {"model": self._model, "prompt": prompt, "stream": False, "think": False, "options": {"temperature": 0}}
        ).encode()
        try:
            raw = self._transport(f"{self._url}/api/generate", body, self._timeout)
        except (OSError, ValueError) as error:
            logger.warning("%s at %s gave no answer: %s", self._model, self._url, error)
            return None
        try:
            text = json.loads(raw)["response"]
        except (ValueError, KeyError, TypeError) as error:
            logger.warning("%s at %s gave an answer that can't be read: %s", self._model, self._url, error)
            return None
        if not isinstance(text, str):
            return None
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or None
