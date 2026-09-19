from typing import Protocol

from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State


class Predictor[Model](Protocol):
    """The prediction task: from a state and the actions every agent takes at once, the possible outcomes, each with
    its probability, an end state carrying each player's payoff. Any model can fill it — a ruleset's effects rules, a
    lookup table, a decision tree, a DNN — run by a stateless service given the model."""

    def predict(self, model: Model, state: State, joint: JointAction) -> OutcomeDistribution: ...
