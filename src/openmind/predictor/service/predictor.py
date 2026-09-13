import logging

from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.expression import Expression
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.effect import Effect
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.model.when import When
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Predictor:
    """Gives the outcome probability distribution of an action in a state, from a domain's transitions."""

    def __init__(
        self,
        interpreter: Interpreter,
        variable_name_mapper: VariableNameMapper,
        expression_text_mapper: ExpressionTextMapper,
        action_text_mapper: ActionTextMapper,
    ) -> None:
        self._interpreter = interpreter
        self._variable_name_mapper = variable_name_mapper
        self._expression_text_mapper = expression_text_mapper
        self._action_text_mapper = action_text_mapper

    def predict(self, model: TransitionModel, state: State, action: Action) -> OutcomeDistribution:
        transition = next(
            (transition for transition in model.transitions if transition.action == action.name), None
        )
        if transition is None:
            raise KeyError(f"No transition for action {action.name!r}")
        outcomes = tuple(
            (self._apply(branch.effects, state, action), branch.probability) for branch in transition.branches
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.to_text(action),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(outcomes)

    def _apply(self, effects: tuple[Effect, ...], state: State, action: Action) -> State:
        for effect in effects:
            match effect:
                case Assign(target, value):
                    state = self._assign(target, value, state, action)
                case When(condition, then, otherwise):
                    holds = self._interpreter.evaluate(condition, state, action)
                    if not isinstance(holds, bool):
                        raise TypeError(f"Condition gave {holds!r} instead of true or false: {condition!r}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "When %s: %s",
                            self._expression_text_mapper.to_text(condition),
                            "true" if holds else "false",
                        )
                    state = self._apply(then if holds else otherwise, state, action)
                case _:
                    raise TypeError(f"Not an effect: {effect!r}")
        return state

    def _assign(self, target: StateVariable, value: Expression, state: State, action: Action) -> State:
        indices = tuple(self._interpreter.evaluate(index, state, action) for index in target.indices)
        name = self._variable_name_mapper.to_name(target.base, indices)
        if all(key != name for key, _ in state.variables):
            raise KeyError(f"Unknown state variable: {name!r}")
        new_value = self._interpreter.evaluate(value, state, action)
        logger.debug("Set %s = %r", name, new_value)
        return State(tuple((key, new_value if key == name else old) for key, old in state.variables))
