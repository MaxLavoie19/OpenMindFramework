import logging
from dataclasses import replace

import pytest

from openmind.evaluation.service.evaluator_tests import first_mover_decides
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rule.model.python_rule import PythonRule
from openmind.training.factory.training_factory import create_rule_selector
from openmind.training.model.selection_report import SelectionReport
from openmind.training.model.selection_settings import SelectionSettings

pytestmark = pytest.mark.log_level("INFO")

DOMAIN = "first mover decides"
A_TO_PLAY = PythonRule("turn == 'A'")
SETTINGS = SelectionSettings(
    positions=None,
    reference_iterations=None,
    iterations=1,
    guided_rollouts=True,
    margin=0.005,
    confidence=0.95,
    resamples=200,
    seed=1,
    max_hours=None,
)
#: Without other rules, lose would be rated above win: with 1 iteration, the search plays the top-rated action.
BASES = (Rule("win", (), 0.0, 10), Rule("lose", (), 0.9, 10), Rule("tie", (), 0.5, 10))
WIN_IS_BEST = Rule("win", (A_TO_PLAY,), 1.0, 5)
LOSE_IS_WORST = Rule("lose", (A_TO_PLAY,), 0.0, 5)
NEVER_MATCHES = Rule("tie", (PythonRule("turn == 'B'"),), 0.2, 5)


def test_rules_play_can_do_without_are_removed_and_the_selection_confirmed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    candidates = RuleBase(DOMAIN, (*BASES, WIN_IS_BEST, LOSE_IS_WORST, NEVER_MATCHES))
    progress: list[SelectionReport] = []

    report = create_rule_selector().select(first_mover_decides(), candidates, "candidates.json", SETTINGS, progress.append)

    assert report.selected.rules == (*BASES, WIN_IS_BEST)
    assert [(test.rule, test.pass_number, test.removed, test.free) for test in report.tests] == [
        (NEVER_MATCHES, 1, True, True),
        (WIN_IS_BEST, 1, False, False),
        (LOSE_IS_WORST, 1, True, False),
        (WIN_IS_BEST, 2, False, False),
    ]
    assert [test.regret_difference for test in report.tests] == [0.0, 0.5, 0.0, 1.0]
    assert (report.complete, report.candidates, report.full.positions, report.subset.mean_regret) == (True, 6, 1, 0.0)
    assert report.confirmation is not None and report.confirmation.holds
    assert len(progress) == 5 and progress[-1] == report
    assert any(message.startswith("Kept win when turn == 'A': EV 1.0 over 5 visits: regret +0.5") for message in caplog.messages)
    assert "Pass 2 removed 0 rules: 4 of 6 kept" in caplog.messages


def test_running_out_of_time_stops_the_passes_and_leaves_the_report_incomplete() -> None:
    candidates = RuleBase(DOMAIN, (*BASES, WIN_IS_BEST, LOSE_IS_WORST))

    report = create_rule_selector().select(
        first_mover_decides(), candidates, "candidates.json", replace(SETTINGS, max_hours=0.0)
    )

    assert (report.complete, report.tests, report.selected) == (False, (), candidates)
    assert report.confirmation is not None


def test_a_reference_search_needs_a_number_of_positions() -> None:
    with pytest.raises(ValueError, match="number of positions"):
        create_rule_selector().select(
            first_mover_decides(), RuleBase(DOMAIN, BASES), "candidates.json", replace(SETTINGS, reference_iterations=10)
        )
