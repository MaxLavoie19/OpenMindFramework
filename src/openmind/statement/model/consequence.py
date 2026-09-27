from dataclasses import dataclass

from openmind.statement.model.drawn import Drawn
from openmind.statement.model.clause import Clause


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
    #: Which action does it. A game of several phases has an action per phase and each does its own thing, so
    #: what an action leads to is not a fact about the game but about that action.
    action: str
    #: Which grid or scalar it changes.
    model: str
    #: Each coordinate of the square it names, drawn from the action. Empty for a scalar.
    where: tuple[Drawn, ...] = ()
    #: Where a move carries something to, drawn the same way. Only a move has one.
    onto: tuple[Drawn, ...] = ()
    #: What is put down, or what a scalar now reads. Nothing for a removal or a move.
    value: Drawn | None = None
    #: The conditions under which it happens, as ways of being so — it happens where any of them covers the
    #: action. They are the same kind of thing as the conditions under which an action is refused, and are
    #: learned by the same machinery.
    when: tuple[Clause, ...] = ()
    #: Where among an action's changes this one was seen, so they can be made in the order the game made them.
    #:
    #: **Order is not presentation here, it is the difference between taking a piece and losing one.** `Changer`
    #: says so itself: what stands somewhere is removed and *then* something moves onto it, which is a capture —
    #: the other way round, the thing that moved is what gets removed. A game shows the order in every sighting
    #: and it was being thrown away, so the consequences came back sorted by nothing in particular. Drawn in that
    #: order, every chess move moved a piece onto a square and then deleted it, and the position a move led to
    #: had a piece missing and no error anywhere.
    order: int = 0
    #: Whether every place of it has narrowed to one drawing, or several still fit what has been seen.
    #:
    #: **Said, because otherwise a guess reads as an answer.** Where more than one drawing survives, one of them
    #: is shown — and a reader has no way to tell that from a drawing the sightings settled. A capture was
    #: reported as being at the column of the mover and the column of the mover stepped across, which is not
    #: where anything was taken; it was the first of several still standing, sorted by name. Thin evidence
    #: looking like a confident error is what sends somebody hunting a fault that is not there.
    settled: bool = True

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
            # Their bodies and not the clauses themselves. They are learned by the machinery that learns what a
            # game refuses, so each carries that head — and "this happens where refused :- …" is not what it
            # says. What holds is the conditions; what they conclude here is that the consequence follows.
            said += ", where " + " or where ".join(
                one.readable.split(" :- ", 1)[-1] for one in self.when
            )
        return said if self.settled else f"{said} — or another way, not narrowed yet"

    def _said(self, where: tuple[Drawn, ...]) -> str:
        return "(" + ", ".join(str(one) for one in where) + ")"
