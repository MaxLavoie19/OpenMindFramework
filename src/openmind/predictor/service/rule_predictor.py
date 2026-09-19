from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.predictor.service.effects_runner import EffectsRunner
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.rule.constant.rule_constant import EFFECTS_DEFINITIONS
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


class RulePredictor:
    """The predictor model that runs an RBS's effects rules: each action's effects, then the rules for what the actions
    together lead to, seeing the definitions its effects see. It keeps nothing: the RBS is given with every call."""

    def __init__(self, effects_runner: EffectsRunner) -> None:
        self._runner = effects_runner

    def predict(self, rbs: RuleBasedSystem, state: State, joint: JointAction) -> OutcomeDistribution:
        """The outcomes of the actions taken at once, each seeing its player as `player`."""
        effects = {action.name: rbs.effects(action.name) for _, action in joint.actions}
        return self._runner.run_joint(state, joint, effects, rbs.effects(None), rbs.definitions(EFFECTS_DEFINITIONS))

    def predict_action(self, rbs: RuleBasedSystem, state: State, action: Action) -> OutcomeDistribution:
        """The outcomes of one action, run without a player, for callers that don't know who takes it. Without an
        effects rule for it, raises KeyError."""
        return self._runner.run(state, action, rbs.effects(action.name), rbs.definitions(EFFECTS_DEFINITIONS))
