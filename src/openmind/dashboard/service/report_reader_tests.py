from dataclasses import replace
from pathlib import Path

from openmind.dashboard.service.report_reader import ReportReader
from openmind.training.mapper.training_report_json_mapper import TrainingReportJsonMapper
from openmind.training.mapper.training_report_json_mapper_tests import REPORT, ROUND
from openmind.training.model.pondering_summary import PonderingSummary


def test_the_newest_report_gives_a_row_per_round_and_the_latest_rules(tmp_path: Path) -> None:
    report = replace(
        REPORT,
        rounds=(ROUND, replace(ROUND, number=2, baselines=(), against_previous=None, pondering=PonderingSummary(50, 7, 12, 3, 1))),
    )
    (tmp_path / "2026-09-15_08-54-52.json").write_text(TrainingReportJsonMapper().to_json(report), encoding="utf-8")

    summary = ReportReader().summary(tmp_path)

    assert summary is not None
    assert (summary.created_at, summary.complete, len(summary.rounds)) == ("2026-09-14T13:00:00", False, 2)
    first, second = summary.rounds
    assert (first.rules, first.held_out_loss, first.baselines, first.against_previous) == (
        1,
        0.691229,
        (("random", "12 / 8 / 0"), ("untrained MCTS", "0 / 20 / 0")),
        "start rules: 5 / 10 / 5",
    )
    assert (second.baselines, second.against_previous, second.pondering) == ((), None, (50, 7, 12, 3, 1))
    assert summary.latest_rules == (("wins(me)", 0.7),)


def test_a_directory_without_reports_gives_nothing(tmp_path: Path) -> None:
    assert ReportReader().summary(tmp_path) is None
