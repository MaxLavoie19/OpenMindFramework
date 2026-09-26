from collections.abc import Mapping, Sequence
from typing import Protocol

from openmind.rule.model.consequence import Consequence
from openmind.structure.model.value import Value
from openmind.world.model.state import State


class ConsequenceCaller(Protocol):
    """What makes learned consequences happen where an effects rule is asked for: the position they leave.

    The other half of `ClauseCaller`, and the same argument. Turning consequences into Python would be a second
    answer to a question already answered — drawing a change from an action is what the predictor does to
    predict, so one mechanism can serve predicting and playing, and what an action does cannot come to mean one
    thing while it is being learned and another once it is used.

    **Nothing is given here that a game would not have.** Which player is acting is read off the position, as
    the readings read it, and which players there are is the game's own. A consequence whose conditions the
    position does not meet does not happen, which is what keeps a learned effects rule from saying that every
    move takes something."""

    def after(
        self, consequences: Sequence[Consequence], state: State, parameters: Mapping[str, Value], action: str
    ) -> State: ...
