from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchResults:
    """A series of games against one opponent, counted from the evaluated agent's side."""

    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
