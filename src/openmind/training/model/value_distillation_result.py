from dataclasses import dataclass

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_fit import ValueFit
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.pondering_summary import PonderingSummary
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord


@dataclass(frozen=True, slots=True)
class ValueDistillationResult:
    """The value base, every price's fit and the chosen one, the candidate terms, how many rows the rules were fitted on
    and held out, the mean absolute difference between the rules' values and the held-out rows' targets (None without a
    held-out row the rules could value), what pondering gave (None without pondering), and, with the signals target,
    the signal library updated and the records of the signals followed, in the order followed; `records` holds what the
    domain records of the round's games, in the order played, for the round's own file."""

    value_base: ValueBase
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
    training_rows: int
    held_out_rows: int
    held_out_error: float | None
    pondering: PonderingSummary | None = None
    library: SignalLibrary | None = None
    arms: tuple[SignalRecord, ...] = ()
    records: tuple[str, ...] = ()
