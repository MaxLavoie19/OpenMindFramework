import logging

from openmind.heuristic.model.node import Node
from openmind.knowledge.constant.rule_kind_constant import OPTIMUM
from openmind.knowledge.model.policy import Policy
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import RULES_DEFINITIONS
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What an optimum rule reads the policy's sub-goal by, and the player it optimizes for.
SUB_GOAL = "sub_goal"
PLAYER = "player"


class RuleOptimizer:
    """The optimizer that runs a ruleset computing the action: equations giving a firing solution, a controller holding
    a variable on target, whatever the rules say.

    An optimum rule reads the state, the player it optimizes for and the policy's sub-goal, and gives the action: an
    `Action`, or its name with its parameters, such as `("fire", {"angle": 41.3})`. It knows the goal, not the rules,
    so what it gives is checked against the game's constraints; an action the constraints refuse is no solution."""

    def __init__(self, rule_caller: RuleCaller) -> None:
        self._rule_caller = rule_caller

    def optimize(self, model: RuleBasedSystem, policy: Policy, node: Node, player: str) -> Action | None:
        """The action the ruleset computes, or None where it computes none or the constraints refuse it."""
        rules = model.of(OPTIMUM)
        if not rules:
            return None
        parameters = {SUB_GOAL: policy.sub_goal, PLAYER: player}
        computed = self._rule_caller.value(rules[0].rule, node.state, parameters, None, model.definitions(RULES_DEFINITIONS))
        action = self._action(computed)
        if action is None:
            return None
        if not self._allowed(node, action, player):
            logger.debug("The rules computed %s, which the game's constraints refuse", action.name)
            return None
        logger.debug("The rules computed %s for %s", action.name, policy.sub_goal or "the default policy")
        return action

    def _action(self, computed: object) -> Action | None:
        """What the rule gave as an action: an `Action`, or a name with its parameters."""
        if isinstance(computed, Action):
            return computed
        if isinstance(computed, tuple) and len(computed) == 2 and isinstance(computed[0], str):
            name, parameters = computed
            return Action(name, tuple(sorted(dict(parameters).items())))  # type: ignore[arg-type]
        return None

    def _allowed(self, node: Node, action: Action, player: str) -> bool:
        return bool(node.game.allows(node.state, action, player=player))  # type: ignore[union-attr]
