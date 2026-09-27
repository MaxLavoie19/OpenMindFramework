from collections.abc import Mapping
from dataclasses import dataclass, field

from openmind.statement.model.literal import Literal


@dataclass(frozen=True, slots=True)
class Example:
    """One case to learn from: everything that was read of it, and whether the thing being learned held of it.

    `literals` is the case itself, said outright — every reading of it as a ground literal. Learning from cases
    written out this way rather than from one shared store of facts is what suits a game: what was true of *this*
    move is a different set of things from what was true of the last one, and there is no position in which they
    are all true together.

    `holds` is what is being accounted for: that the move was legal, that the outcome came about, that this player
    chose it. The learner never knows which of those it is doing.

    `where` is the position it was read in. Two cases from one position are not two pieces of evidence for a rule
    the way two cases from two positions are — a rule that holds across positions has been tried, and one that
    holds twice in the same position may only have been lucky in that position."""

    literals: tuple[Literal, ...]
    holds: bool
    where: object | None = None

    #: The same readings gathered for looking up, worked out once when the case is made.
    #:
    #: Asking whether a clause covers a case asks after each of its readings in turn, and asking after a reading
    #: by walking every reading of the case turns a question into a search. A case cannot change, so neither can
    #: the way into it.
    held: frozenset[Literal] = field(init=False, compare=False, hash=False, repr=False, default=frozenset())
    by_predicate: Mapping[str, tuple[Literal, ...]] = field(
        init=False, compare=False, hash=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        gathered: dict[str, list[Literal]] = {}
        for literal in self.literals:
            if not literal.negated:
                gathered.setdefault(literal.predicate, []).append(literal)
        object.__setattr__(self, "held", frozenset(self.literals))
        object.__setattr__(self, "by_predicate", {name: tuple(found) for name, found in gathered.items()})

    def says(self, predicate: str) -> tuple[Literal, ...]:
        """What it read under that name."""
        return self.by_predicate.get(predicate, ())
