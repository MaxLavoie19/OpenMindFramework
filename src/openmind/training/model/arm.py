from dataclasses import dataclass

from openmind.knowledge.model.model_record import ModelRecord


@dataclass(frozen=True, slots=True)
class Arm:
    """One heuristic in a ranking: what it is called, and the model it plays with.

    A heuristic may be a ruleset, a network, a lookup table or a mechanism of a few lines; what is ranked is how much
    it is worth to play with, not what it is made of."""

    name: str
    model: ModelRecord


@dataclass(frozen=True, slots=True)
class ArmScore:
    """What a heuristic's games came to: how many it played, and the points they paid it — a win 1, a draw 0.5, a
    loss 0 — with the bound UCB1 put on it when it was last chosen."""

    arm: Arm
    games: int
    points: float
    bound: float | None = None

    @property
    def mean(self) -> float:
        """Its points per game; 0.5 before it has played, which is what an unplayed heuristic is taken to be worth."""
        return self.points / self.games if self.games else 0.5
