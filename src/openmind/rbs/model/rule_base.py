from dataclasses import dataclass

from openmind.rbs.model.rule import Rule


@dataclass(frozen=True, slots=True)
class RuleBase:
    """The rules induced for a domain."""

    domain: str
    rules: tuple[Rule, ...]
