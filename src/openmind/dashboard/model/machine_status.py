from dataclasses import dataclass

from openmind.dashboard.model.process_status import ProcessStatus


@dataclass(frozen=True, slots=True)
class MachineStatus:
    """The machine a training runs on: its memory and swap, total and still available, in bytes; the training's
    processes; and earlyoom's latest lines about sending a process a signal, oldest first."""

    memory_total: int
    memory_available: int
    swap_total: int
    swap_free: int
    processes: tuple[ProcessStatus, ...]
    earlyoom: tuple[str, ...]
