from dataclasses import dataclass

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.pondering_summary import PonderingSummary


@dataclass(frozen=True, slots=True)
class ValueDistillationResult:
    """The value base, every price's fit and the chosen one, the candidate terms, how many rows the rules were fitted on
    and held out, the mean absolute difference between the rules' values and the held-out rows' targets (None without a
    held-out row the rules could value), and what pondering gave (None without pondering)."""

    value_base: ValueBase
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
    training_rows: int
    held_out_rows: int
    held_out_error: float | None
    pondering: PonderingSummary | None = None
