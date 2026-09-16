import time


class WallTimeSource:
    """Real elapsed time, as a robot or an opponent across the board lives it: `time.monotonic`, which the system clock
    being set or adjusted never moves backwards."""

    def now(self) -> float:
        return time.monotonic()
