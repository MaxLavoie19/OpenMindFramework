from dataclasses import dataclass

from openmind.knowledge.model.rule_record import RuleRecord


@dataclass(frozen=True, slots=True)
class Labelling:
    """One way of putting a value on a position, and what it was worth trying: where the values came from, how many
    positions it valued, how many different values it gave, and what it cost in seconds.

    A way that valued nothing, or gave the same value everywhere, taught nothing here. That is what it is for: a
    bootstrapper reads these to stop paying for the same lesson in the next game."""

    source: str
    positions: int
    values: int
    seconds: float

    @property
    def paid(self) -> bool:
        """Whether anything can be fitted on what it gave: several positions, valued differently."""
        return self.positions > 1 and self.values > 1


@dataclass(frozen=True, slots=True)
class Pondering:
    """What came of pondering a game: the heuristics it deduced from the rules alone, and everything needed to judge
    them.

    `rules` are the position rules it settled on, already declared in `context` and registered as a model.
    `tried` holds every way of valuing a position it attempted, paid or not, in the order it tried them. `error` is
    the loss of the chosen fit on positions it was not fitted on, None where nothing was held out."""

    context: str
    rules: tuple[RuleRecord, ...]
    positions: int
    tried: tuple[Labelling, ...]
    seconds: float
    error: float | None = None

    @property
    def deduced(self) -> bool:
        return bool(self.rules)
