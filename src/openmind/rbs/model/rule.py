from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Rule:
    """A rule: for an action with this name, when every condition holds (none means any state), its expected payoff for
    the player to act, measured over this many search visits. Conditions are Python rules giving true or false. A
    priority rule's payoff is near the best or worst seen, and it rates before any other rule."""

    action: str
    conditions: tuple[PythonRule, ...]
    expected_value: float
    visits: int
    priority: bool = False
