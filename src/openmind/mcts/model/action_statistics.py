from dataclasses import dataclass

from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class ActionStatistics:
    """A root action's visits and its mean payoff for the player acting at the root (0.0 when never visited)."""

    action: Action
    visits: int
    mean_payoff: float
