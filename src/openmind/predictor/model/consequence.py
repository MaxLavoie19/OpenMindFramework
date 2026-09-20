from dataclasses import dataclass

from openmind.inference.service.covering_learner import Covering
from openmind.predictor.model.drawn import Drawn


@dataclass(frozen=True, slots=True)
class Consequence:
    """One change an action makes, said in terms of the action rather than of the position it left.

    A position handed back says what became of the board; it does not say what the action did, and it says it for
    one position only. A consequence says the doing: this action removes something, and the square it removes from
    is the row of where it started and the column of where it lands — which holds in every position rather than in
    the one it was read from.

    `when` is the conditions under which it happens, learned the way legality was. A consequence with none happens
    every time the action is played, which is what most of a move is."""

    #: Which kind of change: the name of one of `world.model.change`'s types.
    change: str
    #: Which grid or scalar it changes.
    model: str
    #: Each coordinate of the square it names, drawn from the action. Empty for a scalar.
    where: tuple[Drawn, ...] = ()
    #: Where a move carries something to, drawn the same way. Only a move has one.
    onto: tuple[Drawn, ...] = ()
    #: What is put down, or what a scalar now reads. Nothing for a removal or a move.
    value: Drawn | None = None
    #: The conditions under which it happens, as ways of being so — it happens where any of them covers the action.
    when: tuple[Covering, ...] = ()

    @property
    def readable(self) -> str:
        said = f"{self.change.lower()} {self.model}"
        if self.where:
            said += f" at {self._said(self.where)}"
        if self.onto:
            said += f" onto {self._said(self.onto)}"
        if self.value is not None:
            said += f", holding {self.value}"
        if self.when:
            said += ", where " + " or where ".join(one.readable for one in self.when)
        return said

    def _said(self, where: tuple[Drawn, ...]) -> str:
        return "(" + ", ".join(str(one) for one in where) + ")"
