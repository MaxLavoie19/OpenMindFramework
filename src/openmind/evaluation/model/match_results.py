from dataclasses import dataclass

from openmind.timing.model.time_control import TimeControl


@dataclass(frozen=True, slots=True)
class MatchResults:
    """A series of games against one opponent, counted from the evaluated agent's side; on a clock, its time control and
    how many of the wins and losses came from a player's time running out."""

    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    time_control: TimeControl | None = None
    wins_on_time: int = 0
    losses_on_time: int = 0
