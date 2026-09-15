import os

from openmind.parallel.service.memory_guard import MemoryGuard
from openmind.parallel.service.memory_meter import MemoryMeter

_guards: dict[int, MemoryGuard] = {}


def process_memory_guard() -> MemoryGuard:
    """This process's memory guard: memory is held per process, so every cache of a process registers with the same
    guard, and a worker process gets its own."""
    pid = os.getpid()
    guard = _guards.get(pid)
    if guard is None:
        guard = _guards[pid] = MemoryGuard(MemoryMeter())
    return guard
