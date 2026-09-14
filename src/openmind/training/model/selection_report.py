from dataclasses import dataclass
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.rbs.model.rule_base import RuleBase
from openmind.training.model.non_inferiority import NonInferiority
from openmind.training.model.removal_test import RemovalTest
from openmind.training.model.selection_settings import SelectionSettings


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """A rule selection: the domain, when it started, the candidates' file, the settings, how many candidates there were,
    the rules selected so far, how the full and selected sets played on the selection positions, the confirmation
    (None until it runs), every removal tried, and whether the passes ran to the end rather than out of time."""

    domain: str
    created_at: datetime
    candidates_file: str
    settings: SelectionSettings
    candidates: int
    selected: RuleBase
    full: Agreement
    subset: Agreement
    confirmation: NonInferiority | None
    tests: tuple[RemovalTest, ...]
    complete: bool
