import logging

from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.service.rule_caller import RuleCaller
from openmind.world.constant.players_constant import PLAYER
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Predictor:
    """Gives the outcome probability distribution of an action in a state, from a domain's transitions: each branch's
    effects run on the state, with the action's parameters and, for a script, the model's definitions. Actions taken at once
    run one after another, each seeing its player as `player`, then the model's resolution runs."""

    def __init__(self, rule_caller: RuleCaller, action_text_mapper: ActionTextMapper) -> None:
        self._rule_caller = rule_caller
        self._action_text_mapper = action_text_mapper

    def predict(self, model: TransitionModel, state: State, action: Action) -> OutcomeDistribution:
        transition = self._transition(model, action)
        parameters = dict(action.parameters)
        outcomes: list[tuple[State, float]] = []
        for branch in transition.branches:
            outcome = self._rule_caller.apply(branch.effects, state, parameters, model.definitions)
            self._log_changes(state, outcome)
            outcomes.append((outcome, branch.probability))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.to_text(action),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(tuple(outcomes))

    def predict_joint(self, model: TransitionModel, state: State, joint: JointAction) -> OutcomeDistribution:
        """The outcomes of actions taken at once: each player's action runs, in the joint's order, on every outcome so
        far, its effects reading its parameters and its player as `player`, the branches' probabilities multiplying;
        then the model's resolution runs on each outcome. A joint without an action, or an action with a parameter named
        `player`, raises ValueError."""
        if not joint.actions:
            raise ValueError("A joint action needs at least one action")
        outcomes: list[tuple[State, float]] = [(state, 1.0)]
        for player, action in joint.actions:
            transition = self._transition(model, action)
            parameters = dict(action.parameters)
            if PLAYER in parameters:
                raise ValueError(f"Action {action.name!r} has a parameter named {PLAYER!r}, the name of its player")
            parameters[PLAYER] = player
            outcomes = self._branch(model, outcomes, transition.branches, parameters)
        if model.resolution:
            outcomes = self._branch(model, outcomes, model.resolution, None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.joint_text(joint),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(tuple(outcomes))

    def _transition(self, model: TransitionModel, action: Action) -> Transition:
        transition = next((transition for transition in model.transitions if transition.action == action.name), None)
        if transition is None:
            raise KeyError(f"No transition for action {action.name!r}")
        return transition

    def _branch(
        self,
        model: TransitionModel,
        outcomes: list[tuple[State, float]],
        branches: tuple,  # type: ignore[type-arg]
        parameters: dict[str, object] | None,
    ) -> list[tuple[State, float]]:
        branched: list[tuple[State, float]] = []
        for current, probability in outcomes:
            for branch in branches:
                outcome = self._rule_caller.apply(branch.effects, current, parameters, model.definitions)  # type: ignore[arg-type]
                self._log_changes(current, outcome)
                branched.append((outcome, probability * branch.probability))
        return branched

    def _log_changes(self, state: State, outcome: State) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            previous = dict(state.variables)
            for name, after in outcome.variables:
                if name not in previous or (after is not previous[name] and after != previous[name]):
                    logger.debug("Set %s = %r", name, after)
