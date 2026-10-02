from dataclasses import dataclass

from openmind.inference.model.pattern_condition import PatternCondition


@dataclass(frozen=True, slots=True)
class Pattern:
    """Conditions on variables around an index, counted over every index of the anchor base: how many indices `at` meet
    every condition and none of the absences."""

    anchor: str
    conditions: tuple[PatternCondition, ...]
    #: Groups of conditions that must **not** all hold together at the index.
    #:
    #: **The one thing a conjunction cannot say is that something is not there.** `conditions` are joined with
    #: `and` and compare with `==` or `!=`, which says a thing *is* somewhere; a game that keeps what a thing
    #: is and whose it is in two bases then cannot say "no piece of theirs of that kind here", because that is
    #: `not (kind == pawn and owner == them)` — a disjunction, and there is no `or` here.
    #:
    #: Measured against the Chess Intelligence Agent's vocabulary: of its terms that are conjunctions on
    #: paper, seven could be written and nine could not, and all nine failed on exactly this. Passed pawn,
    #: isolated, backward, outpost and hole are all an absence of a particular enemy piece over some squares.
    #:
    #: A group of one is not kept, because `!=` already says it. Two is where this starts to buy anything.
    absences: tuple[tuple[PatternCondition, ...], ...] = ()
