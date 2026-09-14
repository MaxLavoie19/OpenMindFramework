import json
from datetime import datetime

from openmind.evaluation.model.agreement import Agreement
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rule.model.python_rule import PythonRule
from openmind.training.mapper.selection_report_json_mapper import SelectionReportJsonMapper
from openmind.training.model.non_inferiority import NonInferiority
from openmind.training.model.removal_test import RemovalTest
from openmind.training.model.selection_report import SelectionReport
from openmind.training.model.selection_settings import SelectionSettings

BASE = Rule("place", (), 0.5, 100)
WINS = Rule("place", (PythonRule("win_chance(action) >= 1"),), 1.0, 40, priority=True)
UNUSED = Rule("place", (PythonRule("cell[1, 1] == 'Z'"),), 0.1, 10)


def report(complete: bool = True, confirmation: NonInferiority | None = NonInferiority(4520, 0.0004, 0.0021, 0.005, True)) -> SelectionReport:
    return SelectionReport(
        "tictactoe",
        datetime(2026, 9, 13, 23, 45, 0),
        "data/rbs/tictactoe/candidates.json",
        SelectionSettings(None, None, 10, True, 0.005, 0.95, 10000, 1, 2.0),
        3,
        RuleBase("tictactoe", (BASE, WINS)),
        Agreement(10, 4520, 4212, 0.695, 0.040, 0.0027),
        Agreement(10, 4520, 4208, 0.694, 0.041, 0.0021),
        confirmation,
        (
            RemovalTest(UNUSED, 1, 0.0, 0.0, 0, True, True, 0.0),
            RemovalTest(WINS, 1, 0.052, 0.061, -330, False, False, 4.8),
        ),
        complete,
    )


def test_a_report_maps_to_json_with_rules_as_text() -> None:
    assert json.loads(SelectionReportJsonMapper(RuleTextMapper()).to_json(report())) == {
        "domain": "tictactoe",
        "created_at": "2026-09-13T23:45:00",
        "candidates_file": "data/rbs/tictactoe/candidates.json",
        "settings": {
            "positions": None,
            "reference_iterations": None,
            "iterations": 10,
            "guided_rollouts": True,
            "margin": 0.005,
            "confidence": 0.95,
            "resamples": 10000,
            "seed": 1,
            "max_hours": 2.0,
        },
        "candidates": 3,
        "complete": True,
        "selected": [
            "place in any state: EV 0.5 over 100 visits",
            "place when win_chance(action) >= 1: EV 1.0 over 40 visits, priority",
        ],
        "full": {
            "iterations": 10,
            "positions": 4520,
            "optimal": 4212,
            "optimal_visit_share": 0.695,
            "mean_regret": 0.04,
            "seconds_per_choice": 0.0027,
        },
        "subset": {
            "iterations": 10,
            "positions": 4520,
            "optimal": 4208,
            "optimal_visit_share": 0.694,
            "mean_regret": 0.041,
            "seconds_per_choice": 0.0021,
        },
        "confirmation": {
            "positions": 4520,
            "regret_difference": 0.0004,
            "upper_bound": 0.0021,
            "margin": 0.005,
            "holds": True,
        },
        "tests": [
            {
                "rule": "place when cell[1, 1] == 'Z': EV 0.1 over 10 visits",
                "pass": 1,
                "regret_difference": 0.0,
                "upper_bound": 0.0,
                "optimal_difference": 0,
                "removed": True,
                "free": True,
                "seconds": 0.0,
            },
            {
                "rule": "place when win_chance(action) >= 1: EV 1.0 over 40 visits, priority",
                "pass": 1,
                "regret_difference": 0.052,
                "upper_bound": 0.061,
                "optimal_difference": -330,
                "removed": False,
                "free": False,
                "seconds": 4.8,
            },
        ],
    }


def test_a_report_in_progress_has_no_confirmation() -> None:
    document = json.loads(SelectionReportJsonMapper(RuleTextMapper()).to_json(report(False, None)))

    assert (document["complete"], document["confirmation"]) == (False, None)
