from typing import Self

from openmind.expression.service.interpreter import Interpreter
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_rater import RuleRater
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class RuleRaterBuilder:
    """Sets the rule base a rater rates with and wires its interpreter."""

    def __init__(self) -> None:
        self._rule_base: RuleBase | None = None

    def with_rule_base(self, rule_base: RuleBase) -> Self:
        self._rule_base = rule_base
        return self

    def build(self) -> RuleRater:
        if self._rule_base is None:
            raise ValueError("Rule rater needs a rule base")
        return RuleRater(self._rule_base, Interpreter(VariableNameMapper()))
