from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport


class ReportTextMapper:
    """Maps an evaluation report to readable text: the results against each baseline, then agreement with perfect play
    as a table, guided and unguided side by side when both were measured, then the rules alone."""

    def to_text(self, report: EvaluationReport) -> str:
        lines = [
            f"Against {results.opponent}: {results.games} games, {results.wins} wins, {results.draws} draws, "
            f"{results.losses} losses"
            for results in report.baselines
        ]
        if not report.agreement:
            lines.append("Agreement with perfect play skipped: no positions")
            return "\n".join(lines)
        positions = report.agreement[0].positions
        header = ("iterations", "optimal", "visits on optimal", "mean regret", "seconds per choice")
        if report.unguided_agreement:
            lines.append(
                f"Guided by {report.rules_file} against unguided, on the same {positions} positions "
                f"(every action optimal in {report.every_action_optimal}):"
            )
            rows = [
                (str(guided.iterations), *(f"{first} / {second}" for first, second in zip(self._cells(guided), self._cells(unguided), strict=True)))
                for guided, unguided in zip(report.agreement, report.unguided_agreement, strict=True)
            ]
        else:
            lines.append(
                f"Agreement with perfect play on {positions} positions "
                f"(every action optimal in {report.every_action_optimal}):"
            )
            rows = [(str(agreement.iterations), *self._cells(agreement)) for agreement in report.agreement]
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
        lines.extend("  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in (header, *rows))
        if report.rater is not None:
            rater = report.rater
            lines.append(
                f"Rules alone: ratings separate actions in {rater.distinguishing} of {rater.positions} positions; "
                f"a top-rated action is optimal in {rater.optimal:.1f} of {rater.positions}; "
                f"mean regret {rater.mean_regret:.3f}"
            )
        return "\n".join(lines)

    def _cells(self, agreement: Agreement) -> tuple[str, str, str, str]:
        return (
            str(agreement.optimal),
            f"{agreement.optimal_visit_share:.3f}",
            f"{agreement.mean_regret:.3f}",
            f"{agreement.seconds_per_choice:.6f}",
        )
