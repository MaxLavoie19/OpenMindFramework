from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Goal:
    """Something an agent wants in a context: winning, teaching, staying alive. What an outcome is worth is its worth
    on every goal, weighed by the preferences held for them."""

    name: str
    context: str
    tags: Tags = ()
    id: str = ""
