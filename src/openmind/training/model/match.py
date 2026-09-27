from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Match:
    """What came of playing two models of the same task against each other: how many games, and what share of
    the points on offer each took.

    `points` is a win for one, a draw for a half, a loss for none — the middle case allowed to exist, because
    a drawn game is evidence and calling it a loss for both would be a claim neither game made.

    `sides` is how the games were split between the two seatings — the first count with `one` in the first
    player's seat, the second with `other` there. Both models played every game, one on each side of it, so
    what this says is whether either had the first move more often. An odd number of games cannot be split
    evenly and nothing here pretends otherwise: it reports the split and leaves the reading to whoever reads
    it."""

    context: str
    task: str
    one: str
    other: str
    games: int
    points: tuple[float, float]
    sides: tuple[int, int]

    @property
    def decisive(self) -> bool:
        """Whether the two came out apart at all. Equal points over any number of games is a match that chose
        nothing, which is worth saying rather than rounding into a winner."""
        return self.points[0] != self.points[1]

    @property
    def winner(self) -> str:
        """Whichever took more points, or empty where they finished level."""
        if not self.decisive:
            return ""
        return self.one if self.points[0] > self.points[1] else self.other
