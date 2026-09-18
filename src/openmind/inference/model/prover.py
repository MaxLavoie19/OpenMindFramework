from typing import Protocol

from openmind.inference.model.goal import Goal
from openmind.inference.model.proof import Proof


class Prover(Protocol):
    """Proves goals within a time budget."""

    def prove(self, goal: Goal, seconds: float) -> Proof: ...
