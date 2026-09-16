from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelScore:
    """What a model's games came to, as the knowledge base remembers them: its name and id, the sides it played, its
    wins, draws and losses, and when its latest game was remembered."""

    name: str
    id: str
    games: int
    wins: int
    draws: int
    losses: int
    last_game: str

    @property
    def score(self) -> float | None:
        """Points per game, a win 1 and a draw 0.5; None before any game."""
        return None if not self.games else (self.wins + self.draws / 2) / self.games
