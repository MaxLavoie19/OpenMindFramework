from dataclasses import dataclass

from openmind.inference.model.example import Example
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Surprise:
    """A rule that held until it didn't: what it said, what broke it, and how long it had stood.

    A rule broken by the fifth action it met was a guess. One broken after three thousand is a rule of the game that
    almost never comes up — en passant, castling, a pawn reaching the last rank — and meeting it is the most
    informative thing that has happened. How long it stood is what tells the two apart, so it is kept.

    The position is kept too. What is rare is hard to come by again, and every rule deduced afterwards should be made
    to face it rather than wait for chance to bring it back."""

    rule: str
    stood: int
    state: State | None = None
    action: Action | None = None
    readings: tuple[tuple[str, Value], ...] = ()

    #: The case that broke it, where one clause was broken by one case.
    broken_by: Example | None = None

    def stood_longer_than(self, standing: int) -> bool:
        """Whether it is the kind of break worth going back to: one that took long enough to find.

        How long is long enough is the caller's to say and is not a number kept here. What counts as a long
        standing depends on how much has been looked at and on how often the game offers the rare thing, and a
        number chosen here would be a guess about both."""
        return self.stood >= standing
