from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.mcts.model.chance_node import ChanceNode


@dataclass(slots=True)
class SimultaneousNode:
    """A state in the search tree where players act at once. Mutable: it changes while searching.

    For each player to act, in the order of `players` (indices in the players' names): its legal actions, its
    cumulative regrets, its summed strategies, and each action's visits and summed payoffs for that player. `predicted`
    holds, by that same position, the strategies a theory of mind gave other players, which they play instead of regret
    matching. `actions` is empty when the game is over.
    """

    state: State
    players: tuple[int, ...]
    actions: tuple[tuple[Action, ...], ...]
    regrets: list[list[float]]
    strategy_sums: list[list[float]]
    payoff_sums: list[list[float]]
    counts: list[list[int]]
    children: dict[JointAction, ChanceNode]
    visits: int
    predicted: dict[int, tuple[float, ...]] = field(default_factory=dict)
