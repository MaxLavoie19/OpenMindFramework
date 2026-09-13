from datetime import datetime
from pathlib import Path

from openmind.evaluation.mapper.report_json_mapper import ReportJsonMapper
from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.match_results import MatchResults
from openmind.evaluation.repository.report_repository import ReportRepository


def test_save_writes_the_report_under_its_domain_and_time(tmp_path: Path) -> None:
    report = EvaluationReport(
        "tictactoe",
        datetime(2026, 9, 13, 15, 30, 0),
        None,
        EvaluationSettings(games=2, iterations=10, positions=1, budgets=(10,), seed=1),
        (MatchResults("random", 2, 2, 0, 0),),
        0,
        (Agreement(10, 1, 1, 1.0, 0.0, 0.01),),
        (),
        None,
    )

    path = ReportRepository(ReportJsonMapper()).save(report, tmp_path)

    assert path == tmp_path / "tictactoe" / "2026-09-13_15-30-00.json"
    assert path.read_text(encoding="utf-8") == ReportJsonMapper().to_json(report) + "\n"
