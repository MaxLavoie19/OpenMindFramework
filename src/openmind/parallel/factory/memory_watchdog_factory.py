from openmind.parallel.service.memory_watchdog import MemoryWatchdog


def create_memory_watchdog(budget_bytes: int) -> MemoryWatchdog:
    """What a whole run may hold, across the process leading it and every worker it spawned.

    There is no default budget. How much a machine may give a run is a fact about the machine and about who
    else is using it, and neither is knowable here."""
    return MemoryWatchdog(budget_bytes)
