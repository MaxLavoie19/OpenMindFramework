from dataclasses import dataclass
from datetime import datetime

from openmind.knowledge.model.source import Source
from openmind.knowledge.model.tags import Tags
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class DirectExperience:
    """Raw data as received, kept word for word: a text, a clip, a reading, a response from an API, a finished game.
    Nothing rewrites it. It has the shape of a belief without support or certainty: its source says which camera,
    microphone, API, user or referee it came from, and data drawn from it are beliefs whose sources reference it.
    `value` is the raw data, or a reference to where it is stored. `id` is empty until the knowledge base keeps it."""

    variable: str
    context: str
    value: Value
    source: Source
    at: datetime | None = None
    tags: Tags = ()
    id: str = ""
