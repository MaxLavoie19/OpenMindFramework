import json
from datetime import datetime

from openmind.evaluation.mapper.report_json_mapper import ReportJsonMapper
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults


def test_to_json_holds_every_measure() -> None:
    report = EvaluationReport(
        "tictactoe",
        datetime(2026, 9, 13, 15, 30, 0),
        "data/rbs/tictactoe/2026-09-13_15-00-00.json",
        EvaluationSettings(games=4, iterations=50, positions=10, budgets=(10, 50), seed=3),
        (MatchResults("random", 4, 3, 1, 0), MatchResults("untrained MCTS", 4, 0, 4, 0)),
        (Agreement(10, 10, 6, 0.25), Agreement(50, 10, 9, 1.5)),
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
        "agreement": [
            {"iterations": 10, "positions": 10, "optimal": 6, "seconds_per_choice": 0.25},
            {"iterations": 50, "positions": 10, "optimal": 9, "seconds_per_choice": 1.5},
        ],
    }
