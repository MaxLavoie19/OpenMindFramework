import logging

from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Predictor:
    """Gives the outcome probability distribution of an action in a state, from a domain's transitions: each branch's
    effects script runs on the state, with the action's parameters and the model's definitions."""

    def __init__(
        self, rule_compiler: RuleCompiler, rule_runner: RuleRunner, action_text_mapper: ActionTextMapper
    ) -> None:
        self._rule_compiler = rule_compiler
        self._rule_runner = rule_runner
        self._action_text_mapper = action_text_mapper

    def predict(self, model: TransitionModel, state: State, action: Action) -> OutcomeDistribution:
        transition = next(
            (transition for transition in model.transitions if transition.action == action.name), None
        )
        if transition is None:
            raise KeyError(f"No transition for action {action.name!r}")
        parameters = dict(action.parameters)
        outcomes: list[tuple[State, float]] = []
        for branch in transition.branches:
            effects = self._rule_compiler.compile_effects(branch.effects, model.definitions)
            outcome = self._rule_runner.apply(effects, state, parameters)
            if logger.isEnabledFor(logging.DEBUG):
                for (name, before), (_, after) in zip(state.variables, outcome.variables, strict=True):
                    if after is not before and after != before:
                        logger.debug("Set %s = %r", name, after)
            outcomes.append((outcome, branch.probability))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.to_text(action),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(tuple(outcomes))
