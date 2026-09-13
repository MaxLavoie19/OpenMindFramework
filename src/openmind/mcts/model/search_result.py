from dataclasses import dataclass

from openmind.mcts.model.action_sample import ActionSample
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The player acting at the root, every root action's statistics, the most visited action, and every tree sample."""

    player: str
    statistics: tuple[ActionStatistics, ...]
    chosen: Action
    samples: tuple[ActionSample, ...]
