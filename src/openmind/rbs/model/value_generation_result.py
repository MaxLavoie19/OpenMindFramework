from dataclasses import dataclass

from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.model.value_fit import ValueFit


@dataclass(frozen=True, slots=True)
class ValueGenerationResult:
    """What fitting a position heuristic left behind: the context its rules were declared under, the rules themselves
    at the chosen price,
    every price's fit in the order fitted, the chosen fit (None when every training payoff is the same and nothing was
    fitted), the candidate terms, and each rule's term with its weight in the chosen fit on standardized values,
    largest first, so strengths compare between terms."""

    context: str
    rules: tuple[RuleRecord, ...]
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
    strengths: tuple[tuple[PythonRule, float], ...] = ()
