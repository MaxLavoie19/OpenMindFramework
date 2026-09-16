import hashlib
from dataclasses import dataclass

from openmind.agent.constant.agent_constant import MODEL_ID_DIGITS


@dataclass(frozen=True, slots=True)
class ModelDescription:
    """A model as it played: `name`, what a person calls it, such as an arm's signal, and `text`, everything needed to
    build it again, word for word. `id` comes from the text alone, so the same model always has the same id and a model
    that changed, a setting or a rule, has another."""

    name: str
    text: str

    @property
    def id(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:MODEL_ID_DIGITS]
