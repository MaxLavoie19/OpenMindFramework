from dataclasses import replace

from openmind.training.mapper.training_report_json_mapper_tests import REPORT, ROUND
from openmind.training.mapper.training_report_text_mapper import TrainingReportTextMapper
from openmind.training.model.pondering_summary import PonderingSummary


def test_each_round_is_a_line_of_the_table() -> None:
    assert TrainingReportTextMapper().to_text(REPORT).splitlines() == [
        "Trained chess value rules for 1 of 2 rounds, not complete; round 1 started from data/values/chess/start.json",
        "round  rules  held-out loss  with no rule  held-out error  against random  against untrained MCTS     against the previous  seconds",
        "    1      1       0.691229      0.691963          0.0110      12 / 8 / 0              0 / 20 / 0  start rules: 5 / 10 / 5     3600",
    ]


def test_a_round_handed_over_before_its_games_shows_none_against_each_opponent() -> None:
    report = replace(REPORT, rounds=(ROUND, replace(ROUND, number=2, baselines=(), against_previous=None)))

    lines = TrainingReportTextMapper().to_text(report).splitlines()

    assert lines[3].split() == ["2", "1", "0.691229", "0.691963", "0.0110", "none", "none", "none", "3600"]


def test_a_round_that_pondered_shows_its_positions_proofs_and_seeds() -> None:
    report = replace(REPORT, rounds=(replace(ROUND, pondering=PonderingSummary(50, 7, 12, 3, 1)),))

    lines = TrainingReportTextMapper().to_text(report).splitlines()

    assert "pondered: proven / seeds / kept / in rules" in lines[1]
    assert "50: 7 / 12 / 3 / 1" in lines[2]


def test_a_training_without_rounds_is_only_its_heading() -> None:
    report = replace(REPORT, rounds=(), settings=replace(REPORT.settings, start_file=None))

    assert TrainingReportTextMapper().to_text(report) == (
        "Trained chess value rules for 0 of 2 rounds, not complete; round 1 started from no value rules"
    )
