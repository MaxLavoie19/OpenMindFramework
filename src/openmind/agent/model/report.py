from dataclasses import dataclass

from openmind.agent.model.delegation import Delegation
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Report:
    """What a child level gives its parent: where it left its level, what its goal was worth there, what it spent, and
    whether it got there.

    `value` is what the level it left is worth to the player it played as, by that level's own utility; None where
    nothing could value it. `reached` is whether the level ended — the fight won, the file written — rather than the
    budget running out. `task` is the id of the task record the run was kept under, so what delegating to that level
    brings can be looked back on."""

    delegation: Delegation
    state: State | None
    value: float | None
    seconds: float
    reached: bool
    task: str = ""
