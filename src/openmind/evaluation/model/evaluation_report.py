from dataclasses import dataclass
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.guidance_test import GuidanceTest
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement
from openmind.evaluation.model.value_measure import ValueMeasure


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """What an evaluation measured: results against each baseline; in how many sampled positions every action is
    optimal; agreement at each iteration budget for the agent, guided by the rules in rules_file and valuing positions
    with the value rules in values_file, each None when not used; and, when either was used, agreement for an unguided
    agent on the same positions and the paired tests of the agent against it at each budget, with the rules alone and the
    values alone measured when they were used (otherwise empty and None)."""

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
    values_file: str | None = None
    valuer: ValueMeasure | None = None
