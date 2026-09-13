import logging
from itertools import product
from operator import itemgetter

from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.expression.model.expression import Expression
from openmind.expression.service.interpreter import Interpreter
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Solver:
    """Finds every action whose parameter values satisfy all of its constraints in a state."""

    def __init__(self, interpreter: Interpreter) -> None:
        self._interpreter = interpreter

    def solve(self, problem: Problem, state: State) -> tuple[Action, ...]:
        legal: list[Action] = []
        candidates = 0
        for definition in problem.actions:
            names = [variable.name for variable in definition.variables]
            for values in product(*(variable.domain.values for variable in definition.variables)):
                candidates += 1
                action = Action(definition.name, tuple(sorted(zip(names, values), key=itemgetter(0))))
                failed = self._failed_constraint(definition, state, action)
                if failed is None:
                    logger.debug("Accepted %r", action)
                    legal.append(action)
                else:
                    logger.debug("Rejected %r: constraint is false: %r", action, failed)
        logger.info("%d of %d candidate actions are legal", len(legal), candidates)
        return tuple(legal)

    def _failed_constraint(
        self, definition: ActionDefinition, state: State, action: Action
    ) -> Expression | None:
        for constraint in definition.constraints:
            result = self._interpreter.evaluate(constraint, state, action)
            if not isinstance(result, bool):
                raise TypeError(
                    f"Constraint of {definition.name!r} gave {result!r} instead of true or false: {constraint!r}"
                )
            if not result:
                return constraint
        return None
