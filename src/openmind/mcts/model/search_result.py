from dataclasses import dataclass

from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The player acting at the root, every root action's statistics, and the most visited action."""

    player: str
    statistics: tuple[ActionStatistics, ...]
    chosen: Action
