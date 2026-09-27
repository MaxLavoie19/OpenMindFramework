from dataclasses import dataclass

from openmind.statement.model.consequence import Consequence


@dataclass(frozen=True, slots=True)
class ConsequenceRule:
    """What an action does, kept as the consequences learned of it rather than as code that does it.

    The other half of `ClauseRule`, and here for the same reason. A rule kept as Python is opaque the moment it
    is stored: it can be run, and that is all. Nothing can ask whether one effects rule says more than another,
    resolve two of them, or read one out in words. Kept as consequences it is open to all of those — and it is
    what the predictor learns anyway, so nothing has to be turned into anything.

    **A consequence is a rule and this is where that becomes true.** *This action removes something, and the
    square it removes from is the row of where it started and the column of where it lands* holds in every
    position, which is precisely what makes a clause a rule rather than a fact. Said that way it can be stored,
    argued with, and handed to a game — which is what a learned game needs, because a game that cannot say what
    its actions do cannot be played at all.

    They are held in the order the game made them, which is not presentation: what stands somewhere is removed
    and *then* something moves onto it, which is a capture. The other way round, the thing that moved is what
    gets deleted."""

    consequences: tuple[Consequence, ...] = ()

    @property
    def readable(self) -> str:
        return "; ".join(one.readable for one in self.consequences) or "nothing happens"
