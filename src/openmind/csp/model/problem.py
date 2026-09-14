from dataclasses import dataclass

from openmind.csp.model.action_definition import ActionDefinition
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Problem:
    """A constraint satisfaction problem: the actions a domain allows, and the definitions whose names its constraints
    see (None for none)."""

    actions: tuple[ActionDefinition, ...]
    definitions: PythonRule | None = None
