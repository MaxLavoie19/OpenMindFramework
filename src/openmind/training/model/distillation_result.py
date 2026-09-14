from dataclasses import dataclass

from openmind.rbs.model.coverage import Coverage
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.model.hypothesis_test import HypothesisTest
from openmind.rbs.model.rule_base import RuleBase


@dataclass(frozen=True, slots=True)
class DistillationResult:
    """The generated rule base, how many samples it came from and was validated and measured on, the visit-weighted
    mean absolute difference between its ratings and the held-out search results (None without held-out samples to
    rate), the mean number of conditions per rule, the goal patterns found, every hypothesis tested, and the validated
    rules left out because a simpler rule covers them."""

    rule_base: RuleBase
    training_samples: int
    held_out_samples: int
    rating_error: float | None
    mean_conditions: float
    patterns: tuple[GoalPattern, ...]
    hypotheses: tuple[HypothesisTest, ...]
    covered: tuple[Coverage, ...]
