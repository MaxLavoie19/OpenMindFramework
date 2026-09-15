from dataclasses import dataclass

from openmind.inference.model.deduction import Deduction
from openmind.inference.model.expression import Expression
from openmind.rbs.model.position_row import PositionRow
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Pondering:
    """What pondering gave: the training rows, those of proven positions valued at the proven payoffs; every deduction,
    in the order the positions were pondered; and the seeds they induced, the most often induced first, with their
    sources as rules."""

    rows: tuple[PositionRow, ...]
    deductions: tuple[Deduction, ...]
    seeds: tuple[Expression, ...]
    sources: tuple[PythonRule, ...]
