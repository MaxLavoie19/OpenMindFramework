from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RulesetLink:
    """A rule a ruleset lists, by its id, with what the rule weighs in that ruleset. Every link carries a weight, used or
    not. Weights are linear for now: a rule whose value is x contributes weight · x."""

    rule: str
    weight: float = 1.0
