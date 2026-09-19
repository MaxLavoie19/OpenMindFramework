from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Breakpoint:
    """Stops when a reasoning frame of that kind opens and the condition holds: a Python expression, like a game's rule,
    reading the frame's state by model name as rules see it (a scalar as its value) and `frame` itself, such as
    `frame.kind == "evaluation" and en_passant is not None`. Without a condition, every frame of that kind stops."""

    name: str
    frame: str
    condition: PythonRule | None = None
