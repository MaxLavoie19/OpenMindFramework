from dataclasses import dataclass

from openmind.training.model.signal import Signal


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """How well a signal pointed to the coming winner over every round so far: the anchors where the winner read higher
    (agreements) and those where the loser did (disagreements); and the games an agent following the signal played, won,
    drew and lost."""

    signal: Signal
    agreements: int = 0
    disagreements: int = 0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def accuracy(self) -> float:
        """The share of readings that pointed to the winner; 0.5, a coin flip, before any reading."""
        readings = self.agreements + self.disagreements
        return 0.5 if readings == 0 else self.agreements / readings

    @property
    def reliability(self) -> float:
        """2 × accuracy − 1, at least 0: 1 always pointing to the winner, 0 no better than a coin flip."""
        return max(0.0, 2.0 * self.accuracy - 1.0)
