from collections.abc import Sequence
from pathlib import Path

from openmind.parallel.service.memory_watchdog import MemoryWatchdog


def create_memory_watchdog(
    budget_bytes: int = 0, disk_bytes: int = 0, writing_to: Sequence[Path] = ()
) -> MemoryWatchdog:
    """What a whole run may hold across every process it spawned, and how much it may write where it writes.

    Either budget may be nought, which watches that resource not at all; both being nought is refused, since a
    watchdog watching nothing is a misconfigured run pretending to be a safe one.

    There are no defaults. How much a machine may give a run, in memory or in disk, is a fact about the
    machine and about who else is using it, and neither is knowable here."""
    return MemoryWatchdog(budget_bytes, disk_bytes, writing_to)
