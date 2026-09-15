from dataclasses import dataclass

from openmind.inference.model.deduction import Deduction
from openmind.inference.model.expression import Expression
from openmind.rbs.model.position_row import PositionRow
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.ending_walk import EndingWalk


@dataclass(frozen=True, slots=True)
class Pondering:
    """What pondering gave: the training rows, those of proven positions valued at the proven payoffs; every deduction
    of the positions missed most, in the order they were pondered; the seeds all proofs induced, the most often induced
    first, with their sources as rules; and the walks back from the decisive games' ends."""

    rows: tuple[PositionRow, ...]
    deductions: tuple[Deduction, ...]
    seeds: tuple[Expression, ...]
    sources: tuple[PythonRule, ...]
    walks: tuple[EndingWalk, ...] = ()
