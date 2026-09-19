from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.rbs.model.value_fit import ValueFit


@dataclass(frozen=True, slots=True)
class ValueDistillationResult:
    """The context the position rules were declared under and the rules themselves, every price's fit and the chosen
    one, the candidate terms, how many rows the rules were fitted on and held out, the mean absolute difference between
    the rules' values and the held-out rows' targets (None without a held-out row the rules could value); `records`
    holds what the game records of the round's games, in the order played, for the round's own file."""

    context: str
    rules: tuple[RuleRecord, ...]
    fits: tuple[ValueFit, ...]
    chosen: ValueFit | None
    candidates: tuple[PythonRule, ...]
    training_rows: int
    held_out_rows: int
    held_out_error: float | None
    records: tuple[str, ...] = ()
