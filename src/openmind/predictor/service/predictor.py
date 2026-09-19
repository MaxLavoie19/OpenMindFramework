import logging
from collections.abc import Mapping, Sequence

from openmind.structure.model.scalar import Scalar
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.model.rule import Rule
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.world.constant.players_constant import PLAYER
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: An action's outcomes: each effects rule with the chance it happens.
type Effects = Sequence[tuple[float, Rule]]


class Predictor:
    """Gives the outcome probability distribution of an action in a state, from the effects rules an RBS holds for it:
    each rule runs on the state with the action's parameters and, for a script, the context's definitions, and carries
    the chance that outcome happens. Actions taken at once run one after another, each seeing its player as `player`,
    then the rules for what their choices together lead to run."""

    def __init__(self, rule_caller: RuleCaller, action_text_mapper: ActionTextMapper) -> None:
        self._rule_caller = rule_caller
        self._action_text_mapper = action_text_mapper

    def predict(
        self, state: State, action: Action, effects: Effects, definitions: PythonRule | None = None
    ) -> OutcomeDistribution:
        """Each of the action's outcomes with its probability. Without an effects rule, raises KeyError."""
        if not effects:
            raise KeyError(f"No effects rule for action {action.name!r}")
        parameters = dict(action.parameters)
        outcomes: list[tuple[State, float]] = []
        for probability, rule in effects:
            outcome = self._rule_caller.apply(rule, state, parameters, definitions)
            self._log_changes(state, outcome)
            outcomes.append((outcome, probability))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.to_text(action),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(tuple(outcomes))

    def predict_joint(
        self,
        state: State,
        joint: JointAction,
        effects: Mapping[str, Effects],
        together: Effects = (),
        definitions: PythonRule | None = None,
    ) -> OutcomeDistribution:
        """The outcomes of actions taken at once: each player's action runs, in the joint's order, on every outcome so
        far, its effects reading its parameters and its player as `player`, the chances multiplying; then the rules for
        what their choices together lead to run on each outcome. A joint without an action, or an action with a
        parameter named `player`, raises ValueError; an action with no effects rule raises KeyError."""
        if not joint.actions:
            raise ValueError("A joint action needs at least one action")
        outcomes: list[tuple[State, float]] = [(state, 1.0)]
        for player, action in joint.actions:
            action_effects = effects.get(action.name, ())
            if not action_effects:
                raise KeyError(f"No effects rule for action {action.name!r}")
            parameters = dict(action.parameters)
            if PLAYER in parameters:
                raise ValueError(f"Action {action.name!r} has a parameter named {PLAYER!r}, the name of its player")
            parameters[PLAYER] = player
            outcomes = self._branch(outcomes, action_effects, parameters, definitions)
        if together:
            outcomes = self._branch(outcomes, together, None, definitions)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s gives %d outcome(s) with probabilities %s",
                self._action_text_mapper.joint_text(joint),
                len(outcomes),
                [probability for _, probability in outcomes],
            )
        return OutcomeDistribution(tuple(outcomes))

    def _branch(
        self,
        outcomes: list[tuple[State, float]],
        effects: Effects,
        parameters: dict[str, object] | None,
        definitions: PythonRule | None,
    ) -> list[tuple[State, float]]:
        branched: list[tuple[State, float]] = []
        for current, probability in outcomes:
            for chance, rule in effects:
                outcome = self._rule_caller.apply(rule, current, parameters, definitions)  # type: ignore[arg-type]
                self._log_changes(current, outcome)
                branched.append((outcome, probability * chance))
        return branched

    def _log_changes(self, state: State, outcome: State) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            previous = dict(state.models)
            for name, after in outcome.models:
                if name not in previous or (after is not previous[name] and after != previous[name]):
                    logger.debug("Set %s = %r", name, after.value if isinstance(after, Scalar) else after)
