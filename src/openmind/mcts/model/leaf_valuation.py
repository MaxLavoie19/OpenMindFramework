from dataclasses import dataclass

from openmind.mcts.model.position_valuer import PositionValuer


@dataclass(frozen=True, slots=True)
class LeafValuation:
    """How a valuer ends a search's iterations: after up to rollout_actions rollout actions, a position still in play gets
    the valuer's payoffs instead of being played out to the end; where the valuer knows nothing, the rollout goes on."""

    valuer: PositionValuer
    rollout_actions: int = 0
