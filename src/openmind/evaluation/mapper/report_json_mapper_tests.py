import json
from datetime import datetime

from openmind.evaluation.mapper.report_json_mapper import ReportJsonMapper
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.model.rater_agreement import RaterAgreement


def test_to_json_holds_every_measure() -> None:
    report = EvaluationReport(
        "tictactoe",
        datetime(2026, 9, 13, 15, 30, 0),
        "data/rbs/tictactoe/2026-09-13_15-00-00.json",
        EvaluationSettings(games=4, iterations=50, positions=10, budgets=(10, 50), seed=3),
        (MatchResults("random", 4, 3, 1, 0), MatchResults("untrained MCTS", 4, 0, 4, 0)),
        2,
        (Agreement(10, 10, 6, 0.5, 0.125, 0.25), Agreement(50, 10, 9, 0.75, 0.0625, 1.5)),
        (Agreement(10, 10, 5, 0.25, 0.25, 0.125), Agreement(50, 10, 8, 0.5, 0.125, 1.0)),
        RaterAgreement(10, 4, 5.5, 0.2),
    )

    assert json.loads(ReportJsonMapper().to_json(report)) == {
        "domain": "tictactoe",
        "created_at": "2026-09-13T15:30:00",
        "rules_file": "data/rbs/tictactoe/2026-09-13_15-00-00.json",
        "settings": {"games": 4, "iterations": 50, "positions": 10, "budgets": [10, 50], "seed": 3},
        "baselines": [
            {"opponent": "random", "games": 4, "wins": 3, "draws": 1, "losses": 0},
            {"opponent": "untrained MCTS", "games": 4, "wins": 0, "draws": 4, "losses": 0},
        ],
        "every_action_optimal": 2,
        "agreement": [
            {
                "iterations": 10,
                "positions": 10,
                "optimal": 6,
                "optimal_visit_share": 0.5,
                "mean_regret": 0.125,
                "seconds_per_choice": 0.25,
            },
            {
                "iterations": 50,
                "positions": 10,
                "optimal": 9,
                "optimal_visit_share": 0.75,
                "mean_regret": 0.0625,
                "seconds_per_choice": 1.5,
            },
        ],
        "unguided_agreement": [
            {
                "iterations": 10,
                "positions": 10,
                "optimal": 5,
                "optimal_visit_share": 0.25,
                "mean_regret": 0.25,
                "seconds_per_choice": 0.125,
            },
            {
                "iterations": 50,
                "positions": 10,
                "optimal": 8,
                "optimal_visit_share": 0.5,
                "mean_regret": 0.125,
                "seconds_per_choice": 1.0,
            },
        ],
        "rater": {"positions": 10, "distinguishing": 4, "optimal": 5.5, "mean_regret": 0.2},
    }


def test_an_unguided_report_has_no_unguided_agreement_and_no_rater() -> None:
    report = EvaluationReport(
        "tictactoe",
        datetime(2026, 9, 13, 15, 30, 0),
        None,
        EvaluationSettings(games=0, iterations=50, positions=None, budgets=(10,), seed=3),
        (),
        0,
        (Agreement(10, 4520, 4000, 0.5, 0.125, 0.25),),
        (),
        None,
    )

    document = json.loads(ReportJsonMapper().to_json(report))

    assert (document["settings"]["positions"], document["unguided_agreement"], document["rater"]) == (None, [], None)
