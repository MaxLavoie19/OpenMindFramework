from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.training.constant.training_constant import CONFIRMATION_SEED_OFFSET
from openmind.training.model.selection_report import SelectionReport


class SelectionReportTextMapper:
    """Maps a selection report to readable text: what was selected, how all the rules and the selected ones played, the
    confirmation, and the selected rules."""

    def __init__(self, rule_text_mapper: RuleTextMapper) -> None:
        self._rule_text_mapper = rule_text_mapper

    def to_text(self, report: SelectionReport) -> str:
        removed = [test for test in report.tests if test.removed]
        free = sum(1 for test in removed if test.free)
        passes = max((test.pass_number for test in report.tests), default=0)
        ending = "complete" if report.complete else "stopped at the time limit"
        full, subset = report.full, report.subset
        lines = [
            f"Selected {len(report.selected.rules)} of {report.candidates} candidate rules in {passes} passes, {ending}: "
            f"{len(removed)} removed, {free} of them without a search",
            f"At {full.iterations} iterations on {full.positions} positions: all rules {full.optimal} optimal choices, "
            f"mean regret {full.mean_regret:.4f}; selected rules {subset.optimal} optimal choices, mean regret "
            f"{subset.mean_regret:.4f}",
        ]
        confirmation = report.confirmation
        if confirmation is None:
            lines.append("Not confirmed yet")
        else:
            verdict = "below" if confirmation.holds else "not below"
            lines.append(
                f"Confirmation with seed {report.settings.seed + CONFIRMATION_SEED_OFFSET} on {confirmation.positions} "
                f"positions: regret {confirmation.regret_difference:+.4f}, upper bound {confirmation.upper_bound:.4f}, "
                f"{verdict} the margin {confirmation.margin}"
            )
        lines.append("Selected rules:")
        lines.extend(f"  {self._rule_text_mapper.to_text(rule)}" for rule in report.selected.rules)
        return "\n".join(lines)
