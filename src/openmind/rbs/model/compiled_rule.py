from __future__ import annotations

from dataclasses import dataclass
from types import CodeType

from openmind.rbs.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class CompiledRule:
    """A rule compiled once. A value rule's code is a function taking the parameters it reads, in `arguments` order; an
    effects or definitions rule's code runs as a module. `definitions` is the compiled definitions whose names it sees,
    or None."""

    rule: PythonRule
    kind: str
    code: CodeType
    arguments: tuple[str, ...]
    definitions: CompiledRule | None
