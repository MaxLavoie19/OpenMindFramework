from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LogProgress:
    """Where a training is, from its log: the log, the round being run and how many there are (None before the first
    round starts), what the round's start line says, the self-play games and games against opponents finished, the moves
    searched and the moves deduced during the round, the latest notable lines, oldest first, and how many of the round's
    self-play games finished drawn, every payoff equal, and how many decisive."""

    path: Path
    round: int | None
    rounds: int | None
    round_note: str
    games: int
    matches: int
    searched: int
    deduced: int
    recent: tuple[str, ...]
    draws: int = 0
    decisive: int = 0
