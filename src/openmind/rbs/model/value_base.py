from dataclasses import dataclass

from openmind.rbs.model.value_rule import ValueRule


@dataclass(frozen=True, slots=True)
class ValueBase:
    """The value rules fitted for a domain. A player's value in a position is low + (high - low) × logistic(bias + the
    sum of each rule's weight times its term), low and high being the lowest and highest payoffs seen in training."""

    domain: str
    bias: float
    low: float
    high: float
    rules: tuple[ValueRule, ...]
