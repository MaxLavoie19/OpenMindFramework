import time

from openmind.debug.factory.debugger_factory import process_debugger


class WallTimeSource:
    """Real elapsed time, as a robot or an opponent across the board lives it: `time.monotonic`, which the system clock
    being set or adjusted never moves backwards. Where a debug session freezes OMF's clocks while paused, the time spent
    paused doesn't count."""

    def now(self) -> float:
        return time.monotonic() - process_debugger().frozen_seconds()
