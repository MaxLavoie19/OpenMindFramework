from dataclasses import dataclass

from openmind.inference.model.example import Example
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal


@dataclass(frozen=True, slots=True)
class Pursuit:
    """One candidate the rules allow and the game refuses, and how far the search for what refuses it has got.

    **What a search looks like when it is not in a hurry.** The ordinary learner is given a position's worth of
    unaccounted cases and sixty seconds, so each case gets a shallow look and what comes out is one or two
    conditions. Measured against a run's own constraints, that leaves every error of the same kind: a set
    closing ninety-nine per cent of the space, refusing not one legal move, and letting seventy-five candidates
    through — most of them needing three conditions to refuse, which no shallow look reaches.

    So this is one candidate, worked on by itself for as long as it takes. It carries no services and nothing
    it cannot be sent to another process with: a scheduler keeps it, hands it whatever time there is, and gets
    it back further on than it was.

    **Ends properly rather than on a clock.** A body is worth asking the guard about only where every part of
    it one condition shorter turned away a legal move. So if nothing slipped at one size, every body one longer
    contains a part that already stood, and none of them can beat what stands: the search is finished, at
    whatever depth it reached, with no ceiling and no timeout. `exhausted` says that happened. Five minutes
    then means put it down and come back, never assume it cannot be done."""

    #: The candidate being accounted for: what was read of it, and that the game refused it.
    case: Example
    #: Its readings, tied once when the pursuit began.
    #:
    #: Carried rather than read again, because `at` is a place in the combinations of these in this order. Read
    #: afresh in another session they might come back in another order, and then resuming would skip some
    #: bodies and try others twice.
    offered: tuple[Literal, ...]
    #: How many conditions the bodies being tried have. Zero before it starts.
    size: int = 0
    #: How far into that size's combinations it has got: the place the next session starts from.
    at: int = 0
    #: The best body found so far — most of the pool also refused, then cheapest — or None while none stands.
    found: Clause | None = None
    #: Seconds spent on it, over every session there has been.
    seconds: float = 0.0
    #: Whether the search is finished: nothing slipped at the last size it completed, so no longer body can
    #: beat what stands. With `found` set, that is the best body there is. With `found` still None, it is
    #: something stronger than defeat — nothing the readings can say tells this candidate from the legal
    #: moves, which is a reading the vocabulary has not got.
    exhausted: bool = False

    @property
    def readable(self) -> str:
        where = f"{self.size} conditions, {self.at} in" if not self.exhausted else f"finished at {self.size}"
        return f"{self.found.readable if self.found else 'nothing yet'} ({where}, {self.seconds:.0f}s)"
