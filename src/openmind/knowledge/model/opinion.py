from dataclasses import dataclass
from datetime import datetime

from openmind.knowledge.model.source import Source
from openmind.knowledge.model.tags import Tags
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Opinion:
    """A subjective judgement, the agent's own direct experience, with any qualifier as its value: "good", "cheap", "don't
    like". It may have reasons: "I don't like winter" rests on the opinion "I don't like the cold" and the belief "winter
    is cold", each a source naming it. The agent edits only its own opinions; another agent's opinion is known only as a
    belief held by that agent. `id` is empty until the knowledge base keeps it."""

    variable: str
    context: str
    value: Value
    at: datetime | None = None
    reasons: tuple[Source, ...] = ()
    tags: Tags = ()
    id: str = ""
