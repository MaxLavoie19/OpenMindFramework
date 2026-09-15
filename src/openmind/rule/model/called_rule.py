from dataclasses import dataclass

from openmind.rule.model.compiled_rule import CompiledRule
from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class CalledRule:
    """A rule ready to be called: the rule itself, its compiled code when it is source, the parameters it reads, and how
    it reads in a log. A function reads every parameter it was prepared with, since only source says which it uses."""

    rule: Rule
    compiled: CompiledRule | None
    arguments: tuple[str, ...]
    source: str
