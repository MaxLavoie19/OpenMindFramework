from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.training.mapper.selection_report_json_mapper_tests import report
from openmind.training.mapper.selection_report_text_mapper import SelectionReportTextMapper
from openmind.training.model.non_inferiority import NonInferiority


def test_a_report_reads_as_a_summary_the_confirmation_and_the_selected_rules() -> None:
    assert SelectionReportTextMapper(RuleTextMapper()).to_text(report()).splitlines() == [
        "Selected 2 of 3 candidate rules in 1 passes, complete: 1 removed, 1 of them without a search",
        "At 10 iterations on 4520 positions: all rules 4212 optimal choices, mean regret 0.0400; selected rules 4208 "
        "optimal choices, mean regret 0.0410",
        "Confirmation with seed 2 on 4520 positions: regret +0.0004, upper bound 0.0021, below the margin 0.005",
        "Selected rules:",
        "  place in any state: EV 0.5 over 100 visits",
        "  place when win_chance(action) >= 1: EV 1.0 over 40 visits, priority",
    ]


def test_an_unfinished_or_failed_selection_says_so() -> None:
    mapper = SelectionReportTextMapper(RuleTextMapper())

    assert mapper.to_text(report(False, None)).splitlines()[0].endswith(", stopped at the time limit: 1 removed, 1 of them without a search")
    assert mapper.to_text(report(False, None)).splitlines()[2] == "Not confirmed yet"
    worse = NonInferiority(4520, 0.004, 0.006, 0.005, False)
    assert mapper.to_text(report(True, worse)).splitlines()[2].endswith("upper bound 0.0060, not below the margin 0.005")
