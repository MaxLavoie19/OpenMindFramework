from dataclasses import dataclass

from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class Rule:
    """An induced rule: for an action with this name, when every condition holds (none means any state), its expected
    payoff for the player to act, measured over this many search visits."""

    action: str
    conditions: tuple[Expression, ...]
    expected_value: float
    visits: int
