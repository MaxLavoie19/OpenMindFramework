from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PositionRow:
    """A position valued for a player, by name, and the payoff its value is fitted to."""

    state: State
    player: str
    target: float
    #: How well grounded that target is, as a share of an ordinary row's say in the fit.
    #:
    #: **Because the two things a position can be valued by are not equally known.** A proof is certain: the
    #: rules were followed to a result and the number is what the position *is* worth. A search's verdict is
    #: an estimate — the heuristic's own reading improved by looking ahead, which is better than the reading
    #: and is not the truth. Fitted side by side at equal say, the estimates outnumber the proofs and the fit
    #: learns mostly from the weaker of the two.
    #:
    #: One is an ordinary row. Below one is a row the fit should listen to less, and nought is a row it should
    #: not hear at all. Nothing here says what a searched value is worth against a proof: that is a judgement
    #: about a particular search in a particular game, and it belongs to whoever ran it.
    certainty: float = 1.0
