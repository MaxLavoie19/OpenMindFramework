from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.mcts.model.decision_node import DecisionNode
    from openmind.mcts.model.simultaneous_node import SimultaneousNode


@dataclass(slots=True)
class ChanceNode:
    """An action taken in the search tree, or actions taken at once, with its possible outcomes. Mutable: it changes
    while searching."""

    action: Action | JointAction
    outcomes: tuple[tuple[State, float], ...]
    children: dict[State, DecisionNode | SimultaneousNode]
    visits: int
    payoff_sums: list[float]
