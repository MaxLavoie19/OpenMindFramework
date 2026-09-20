import logging
import math

import numpy as np

from openmind.heuristic.model.node import Node
from openmind.knowledge.constant.rule_kind_constant import POSITION
from openmind.knowledge.model.policy import Policy
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller

logger = logging.getLogger(__name__)

#: What a policy value rule reads the policy by: its name, its sub-goal, and the player it is rated for.
POLICY = "policy"
SUB_GOAL = "sub_goal"
ME = "me"


class RulePolicyValuer:
    """The policy value model that runs a ruleset: each rule reads the state, the policy's name and sub-goal, and says
    what following it here is worth, times its weight in the ruleset, summed.

    A policy takes the utility of the move it decided on, and that utility is what trains such a ruleset."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def rate(
        self, model: RuleBasedSystem, node: Node, policies: tuple[Policy, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each policy is worth following here, in the policies' order; None where no rule could be read."""
        rules = tuple((rule, weight) for rule, weight in model.rules if rule.kind == POSITION)
        if not rules:
            return (None,) * len(policies)
        definitions = model.definitions(RULES_DEFINITIONS)
        rated: list[float | None] = []
        for policy in policies:
            names = {POLICY: policy.name, SUB_GOAL: policy.sub_goal, ME: player}
            readings = []
            for rule, weight in rules:
                try:
                    read = self._rule_caller.value(rule.rule, node.state, None, names, definitions)
                except (KeyError, NameError, TypeError, AttributeError, ValueError, ArithmeticError):
                    continue
                if isinstance(read, bool | int | float | np.bool_ | np.number) and math.isfinite(float(read)):  # type: ignore[arg-type]
                    readings.append(weight * float(read))  # type: ignore[arg-type]
            rated.append(math.fsum(readings) if readings else None)
        return tuple(rated)
