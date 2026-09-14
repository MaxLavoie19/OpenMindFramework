from dataclasses import dataclass

from openmind.rbs.model.rule import Rule


@dataclass(frozen=True, slots=True)
class Coverage:
    """A validated rule left out of the rule base, and the simpler kept rule covering it: one matching every row it
    matches, whose rows' mean advantage is within min_gain of its rows'."""

    rule: Rule
    covering: Rule
