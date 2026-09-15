from collections.abc import Hashable
from dataclasses import dataclass

from openmind.mcts.model.action_statistics import ActionStatistics


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """One hypothesis of a semi-determinized search: its label, the probability the theory of mind gave it, and every
    root action's statistics in the search made as if it were true."""

    label: Hashable
    probability: float
    statistics: tuple[ActionStatistics, ...]
