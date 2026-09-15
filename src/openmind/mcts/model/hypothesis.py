from collections.abc import Hashable
from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A prediction about what a player can't see, such as another player's hidden move: its label, and the states that
    could be true under it with their probabilities, summing to 1."""

    label: Hashable
    completions: tuple[tuple[State, float], ...]
