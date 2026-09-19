from dataclasses import dataclass

from openmind.knowledge.model.tags import Tags


@dataclass(frozen=True, slots=True)
class Mechanism:
    """Whatever produces evidence: a camera, a microphone, an API, a user, a decoder such as "fork detector", an
    inference such as "deduction", self-play, an application's declaration. `declared_accuracy` is the starting
    accuracy whoever declares it gives; None leaves it uninformative."""

    id: str
    name: str
    declared_accuracy: float | None = None
    tags: Tags = ()
