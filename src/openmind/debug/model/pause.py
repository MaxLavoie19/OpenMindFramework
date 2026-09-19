from dataclasses import dataclass

from openmind.debug.model.frame import Frame


@dataclass(frozen=True, slots=True)
class Pause:
    """Why and where a run stopped: the breakpoint or interrupt that stopped it, the reasoning stack, innermost last, and
    the Python stack at that moment as text. A worker process sends it to the process that started it."""

    reason: str
    stack: tuple[Frame, ...]
    python_stack: str
    process: int
