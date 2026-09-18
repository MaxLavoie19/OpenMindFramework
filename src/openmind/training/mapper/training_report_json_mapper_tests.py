import json
from dataclasses import replace
from datetime import datetime

from openmind.evaluation.model.match_results import MatchResults
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.rbs.model.python_rule import PythonRule
from openmind.doxastic.constant.doxastic_constant import COUNTED
from openmind.doxastic.constant.rule_kind_constant import POSITION
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.rule_record import RuleRecord
from openmind.rbs.model.value_fit import ValueFit
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.mapper.training_report_json_mapper import TrainingReportJsonMapper
from openmind.training.model.training_report import TrainingReport
from openmind.training.model.training_round import TrainingRound
from openmind.training.model.value_distillation_settings import ValueDistillationSettings
from openmind.training.model.value_training_settings import ValueTrainingSettings

SETTINGS = ValueTrainingSettings(
    2,
    ValueDistillationSettings(200, 50, 100, 1, "search", ValueSettings((0.1, 0.01), 1000, 1e-6, 3600.0, 8 * 1024**3, 50_000)),
    10,
    100,
    0.5,
    20,
    "data/values/chess/start.json",
)
NO_RULE = ValueFit(0.1, 0, 168, True, 0.691827, 0.691963)
CHOSEN = ValueFit(0.01, 1, 305, True, 0.690585, 0.691229)
ROUND = TrainingRound(
    1,
    "chess",
    (RuleRecord("wins(me)", POSITION, PythonRule("wins(me)"), Provenance(COUNTED, "chess"), (("chess", 0.7),), id="r000001"),),
    (NO_RULE, CHOSEN),
    CHOSEN,
    7211,
    1778,
    0.011,
    (MatchResults("random", 20, 12, 8, 0), MatchResults("untrained MCTS", 20, 0, 20, 0)),
    MatchResults("start rules", 20, 5, 10, 5),
    3600.4,
)
REPORT = TrainingReport("chess", datetime(2026, 9, 14, 13, 0, 0), SETTINGS, (ROUND,), False)


def test_to_json_holds_the_settings_and_every_round() -> None:
    document = json.loads(TrainingReportJsonMapper().to_json(REPORT))

    assert (document["domain"], document["created_at"], document["complete"]) == ("chess", "2026-09-14T13:00:00", False)
    assert document["settings"] == {
        "rounds": 2,
        "start_file": "data/values/chess/start.json",
        "games": 200,
        "held_out_games": 50,
        "iterations": 100,
        "seed": 1,
        "target": "search",
        "time_control": None,
        "expected_steps": 30,
        "time_reserve": 0.05,
        "selection": "ucb1",
        "puct_exploration": 1.5,
        "prior": "uniform",
        "prior_temperature": 0.1,
        "prices": [0.1, 0.01],
        "max_steps": 1000,
        "tolerance": 1e-6,
        "seconds": 3600.0,
        "memory_bytes": 8 * 1024**3,
        "candidates": 50_000,
        "rollout_actions": 10,
        "rollout_limit": 100,
        "unfinished_payoff": 0.5,
        "evaluation_games": 20,
        "deduction": None,
    }
    assert document["rounds"] == [
        {
            "number": 1,
            "context": "chess",
            "position_rules": [{"name": "wins(me)", "weight": 0.7, "id": "r000001"}],
            "fits": [
                {"price": 0.1, "terms_kept": 0, "steps": 168, "settled": True, "training_loss": 0.691827, "held_out_loss": 0.691963},
                {"price": 0.01, "terms_kept": 1, "steps": 305, "settled": True, "training_loss": 0.690585, "held_out_loss": 0.691229},
            ],
            "chosen_price": 0.01,
            "training_rows": 7211,
            "held_out_rows": 1778,
            "held_out_error": 0.011,
            "baselines": [
                {"opponent": "random", "games": 20, "wins": 12, "draws": 8, "losses": 0, "time_control": None, "wins_on_time": 0, "losses_on_time": 0},
                {"opponent": "untrained MCTS", "games": 20, "wins": 0, "draws": 20, "losses": 0, "time_control": None, "wins_on_time": 0, "losses_on_time": 0},
            ],
            "against_previous": {"opponent": "start rules", "games": 20, "wins": 5, "draws": 10, "losses": 5, "time_control": None, "wins_on_time": 0, "losses_on_time": 0},
            "seconds": 3600.4,
        }
    ]


def test_to_json_holds_the_deduction_budget() -> None:
    report = replace(REPORT, settings=replace(SETTINGS, deduction=DeductionBudget(3, 10.0, 1.0)))

    document = json.loads(TrainingReportJsonMapper().to_json(report))

    assert document["settings"]["deduction"] == {"plies": 3, "seconds": 10.0, "highest": 1.0}
