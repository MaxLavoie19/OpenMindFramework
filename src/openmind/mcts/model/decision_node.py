from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openmind.world.model.action import Action
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.mcts.model.chance_node import ChanceNode


@dataclass(slots=True)
class DecisionNode:
    """A state in the search tree where a player picks an action. Mutable: it changes while searching.

    ratings follow the order of actions and are empty when the search isn't guided; priors follow it too, and are empty
    unless the search selects by PUCT.
    """

    state: State
    actions: tuple[Action, ...]
    untried: list[Action]
    children: dict[Action, ChanceNode]
    visits: int
    player: int | None
    ratings: tuple[float, ...]
    priors: tuple[float, ...] = ()
