from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class OutcomeDistribution:
    """Each possible outcome of an action (the new state) with its probability."""

    outcomes: tuple[tuple[State, float], ...]
