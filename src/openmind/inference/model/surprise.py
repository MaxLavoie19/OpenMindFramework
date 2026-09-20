from dataclasses import dataclass

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
    state: State
    action: Action | None = None
    readings: tuple[tuple[str, Value], ...] = ()

    @property
    def rare(self) -> bool:
        """Whether it is the kind of break worth going back to: one that took a long time to find."""
        return self.stood >= 100
