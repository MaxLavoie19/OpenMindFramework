from dataclasses import dataclass
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.guidance_test import GuidanceTest
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """What an evaluation measured: results against each baseline; in how many sampled positions every action is
    optimal; agreement at each iteration budget for the agent, guided by the rules in rules_file or unguided when it's
    None; and, when rules guided it, agreement for an unguided agent on the same positions, the rules alone, and the
    paired tests of guided against unguided at each budget (otherwise empty and None)."""

    domain: str
    created_at: datetime
    rules_file: str | None
    settings: EvaluationSettings
    baselines: tuple[MatchResults, ...]
    every_action_optimal: int
    agreement: tuple[Agreement, ...]
    unguided_agreement: tuple[Agreement, ...]
    rater: RaterAgreement | None = None
    guidance_tests: tuple[GuidanceTest, ...] = ()
