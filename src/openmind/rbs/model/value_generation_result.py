from dataclasses import dataclass

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class ValueGenerationResult:
    """The value base fitted at the chosen price, every price's fit in the order fitted, the chosen fit (None when every
    training payoff is the same and nothing was fitted), and the candidate terms."""

    value_base: ValueBase
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
