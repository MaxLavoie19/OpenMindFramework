from dataclasses import dataclass

from openmind.rbs.model.coverage import Coverage
from openmind.rbs.model.goal_pattern import GoalPattern
from openmind.rbs.model.hypothesis_test import HypothesisTest
from openmind.rbs.model.rule_base import RuleBase


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The rule base kept from generation, the goal patterns found, every hypothesis tested with its result, and the
    validated rules left out because a simpler rule covers them."""

    rule_base: RuleBase
    patterns: tuple[GoalPattern, ...]
    hypotheses: tuple[HypothesisTest, ...]
    covered: tuple[Coverage, ...]
