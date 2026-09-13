from dataclasses import dataclass
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """What an evaluation measured: results against each baseline and agreement at each iteration budget."""

    domain: str
    created_at: datetime
    settings: EvaluationSettings
    baselines: tuple[MatchResults, ...]
    agreement: tuple[Agreement, ...]
