from dataclasses import dataclass
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """What an evaluation measured: results against each baseline and agreement at each iteration budget, for an agent
    guided by the rules in rules_file, or unguided when it's None."""

    domain: str
    created_at: datetime
    rules_file: str | None
    settings: EvaluationSettings
    baselines: tuple[MatchResults, ...]
    agreement: tuple[Agreement, ...]
