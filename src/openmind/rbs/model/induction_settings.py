from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InductionSettings:
    """Which samples count (min_visits), how long rules get (max_conditions), and when a condition is worth adding: the
    narrower rule keeps min_rule_visits visits and its expected value moves by at least min_gain."""

    min_visits: int
    max_conditions: int
    min_rule_visits: int
    min_gain: float
