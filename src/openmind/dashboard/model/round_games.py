from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RoundGames:
    """What a round's self-play games came to, counted from the log as they finished: how many were played, how many
    were decisive and how many drawn, the plies they took together with the shortest and the longest, and how many
    ended each way the domain says games end.

    A round still being played has the games finished so far, so the plots grow with it."""

    number: int
    games: int = 0
    decisive: int = 0
    draws: int = 0
    plies: int = 0
    shortest: int | None = None
    longest: int | None = None
    endings: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def mean_plies(self) -> float:
        """The plies an average game took, 0 before any game has finished."""
        return self.plies / self.games if self.games else 0.0

    @property
    def decisive_share(self) -> float:
        """The share of the games that were decisive, 0 before any game has finished."""
        return self.decisive / self.games if self.games else 0.0
