from dataclasses import dataclass

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class ValueGenerationResult:
    """The value base fitted at the chosen price, every price's fit in the order fitted, the chosen fit (None when every
    training payoff is the same and nothing was fitted), the candidate terms, and each value rule's term with its weight
    in the chosen fit on standardized values, largest first, so strengths compare between terms."""

    value_base: ValueBase
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
    strengths: tuple[tuple[PythonRule, float], ...] = ()
