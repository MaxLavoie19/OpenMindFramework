from dataclasses import dataclass

from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class StateDomain:
    """The values a variable can take in a state: a rule reading the state, and the problem's definitions when it is
    source, that gives them in order; a value given twice counts once."""

    rule: Rule
