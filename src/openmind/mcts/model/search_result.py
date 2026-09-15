from dataclasses import dataclass

from openmind.mcts.model.action_sample import ActionSample
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.hypothesis_result import HypothesisResult
from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The player acting at the root, every root action's statistics, the action chosen (the most visited, in a
    semi-determinized search the highest expected payoff, or where players act at once one sampled from the average
    strategy), every tree sample, a semi-determinized search's hypotheses, and, where players act at once, the searching
    player's average strategy over its root actions."""

    player: str
    statistics: tuple[ActionStatistics, ...]
    chosen: Action
    samples: tuple[ActionSample, ...]
    hypotheses: tuple[HypothesisResult, ...] = ()
    strategy: tuple[tuple[Action, float], ...] = ()
