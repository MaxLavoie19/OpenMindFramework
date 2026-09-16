class ManualTimeSource:
    """Time that moves only when told to, for tests and simulations: it starts at `start` and `advance` moves it on."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        """Moves time on by that many seconds; moving it back raises ValueError, as time never runs backwards."""
        if seconds < 0:
            raise ValueError(f"Time can't move backwards, not by {seconds} seconds")
        self._now += seconds
