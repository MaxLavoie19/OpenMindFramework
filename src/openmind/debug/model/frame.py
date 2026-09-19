from dataclasses import dataclass

from openmind.structure.model.value import Value
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Frame:
    """One step of OMF's reasoning — a task, a tactic, a search, a search node, an evaluation, a rule — with what it
    works on, and where in the Python code it was opened, which links the reasoning stack to the Python one."""

    kind: str
    context: str | None = None
    state: State | None = None
    details: tuple[tuple[str, Value], ...] = ()
    tags: tuple[tuple[str, Value], ...] = ()
    python: str = ""
