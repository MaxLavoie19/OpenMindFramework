from dataclasses import dataclass
from datetime import datetime

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Source:
    """Where something the agent knows came from: the id of the mechanism that produced it, its parameters, when, and
    the ids of the experiences, beliefs, opinions or rules it rests on.

    Direct experience of a game already kept is `Source(<direct experience's id>, rests_on=(<the game's id>,))`; a
    deduction is `Source(<inference's id>, (("method", "deduction"), ("plies", 3)), rests_on=(<premises' ids>))`."""

    mechanism: str
    parameters: tuple[tuple[str, Value], ...] = ()
    at: datetime | None = None
    rests_on: tuple[str, ...] = ()

    def parameter(self, name: str) -> Value:
        """The first value given under that name, or None."""
        return next((value for key, value in self.parameters if key == name), None)
