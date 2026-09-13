from dataclasses import dataclass

from openmind.mcts.model.action_rater import ActionRater


@dataclass(frozen=True, slots=True)
class Guidance:
    """How a rater steers a search: the weight of its ratings in selection and the temperature of rollout choices."""

    rater: ActionRater
    prior_weight: float
    rollout_temperature: float
