from dataclasses import dataclass

from openmind.mcts.model.action_rater import ActionRater


@dataclass(frozen=True, slots=True)
class Guidance:
    """How a rater steers a search: the weight of its ratings in selection, the temperature of rollout choices, and
    whether rollouts follow the ratings at all; without guided rollouts, only the tree's nodes are rated and rollouts
    pick uniformly."""

    rater: ActionRater
    prior_weight: float
    rollout_temperature: float
    guided_rollouts: bool = True
