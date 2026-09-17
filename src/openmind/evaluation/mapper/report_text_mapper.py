from collections.abc import Sequence

from openmind.evaluation.model.agreement import Agreement
from openmind.evaluation.model.evaluation_report import EvaluationReport
from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper


class ReportTextMapper:
    """Maps an evaluation report to readable text: the results against each baseline, then agreement with perfect play
    as a table, guided and unguided side by side when both were measured, then the rules alone, the values alone and the
    paired tests of guided against unguided."""

    def to_text(self, report: EvaluationReport) -> str:
        lines = [
            f"Against {results.opponent}: {results.games} games, {results.wins} wins, {results.draws} draws, "
            f"{results.losses} losses"
            + (
                ""
                if results.time_control is None
                else f", on {TimeControlTextMapper().to_text(results.time_control)}: {results.wins_on_time} wins and "
                f"{results.losses_on_time} losses on time"
            )
            for results in report.baselines
        ]
        if not report.agreement:
            lines.append("Agreement with perfect play skipped: no positions")
            return "\n".join(lines)
        positions = report.agreement[0].positions
        reference = report.settings.reference_iterations
        limit = report.settings.rollout_limit
        detail = (
            f"({'' if reference is None else f'reference: {reference}-iteration unguided searches; '}"
            f"{'' if limit is None else f'rollout limit {limit}, unfinished payoff {report.settings.unfinished_payoff}; '}"
            f"every action optimal in {report.every_action_optimal})"
        )
        header = ("iterations", "optimal", "visits on optimal", "mean regret", "seconds per choice")
        if report.unguided_agreement:
            lines.append(f"{self._agent(report)} against unguided, on the same {positions} positions {detail}:")
            rows = [
                (str(guided.iterations), *(f"{first} / {second}" for first, second in zip(self._cells(guided), self._cells(unguided), strict=True)))
                for guided, unguided in zip(report.agreement, report.unguided_agreement, strict=True)
            ]
        else:
            lines.append(f"Agreement with perfect play on {positions} positions {detail}:")
            rows = [(str(agreement.iterations), *self._cells(agreement)) for agreement in report.agreement]
        lines.extend(self._table(header, rows))
        if report.rater is not None:
            rater = report.rater
            lines.append(
                f"Rules alone: ratings separate actions in {rater.distinguishing} of {rater.positions} positions; "
                f"a top-rated action is optimal in {rater.optimal:.1f} of {rater.positions}; "
                f"mean regret {rater.mean_regret:.3f}"
            )
        if report.valuer is not None:
            valuer = report.valuer
            error = "none" if valuer.mean_absolute_error is None else f"{valuer.mean_absolute_error:.3f}"
            lines.append(
                f"Values alone: valued {valuer.valued} of {valuer.positions} positions, mean absolute error {error}; "
                f"one step ahead, a top-valued action is optimal in {valuer.optimal:.1f} of {valuer.positions}; "
                f"mean regret {valuer.mean_regret:.3f}"
            )
        if report.guidance_tests:
            lines.append("Guided against unguided, paired by position (guided minus unguided; Wilcoxon and McNemar p-values):")
            lines.extend(
                self._table(
                    ("iterations", "low-value visits", "p", "regret", "p", "optimal only guided / unguided", "p"),
                    [
                        (
                            str(test.iterations),
                            f"{test.low_value_share_difference:+.3f}",
                            f"{test.low_value_share_p:.3g}",
                            f"{test.regret_difference:+.3f}",
                            f"{test.regret_p:.3g}",
                            f"{test.optimal_only_guided} / {test.optimal_only_unguided}",
                            f"{test.optimal_choice_p:.3g}",
                        )
                        for test in report.guidance_tests
                    ],
                )
            )
        return "\n".join(lines)

    def _agent(self, report: EvaluationReport) -> str:
        """What guided and valued the agent: Guided by <rules>, Valued by <values>, or both joined by "and"."""
        parts: list[str] = []
        if report.rules_file is not None:
            rollouts = "" if report.settings.guided_rollouts else " with unguided rollouts"
            parts.append(f"guided by {report.rules_file}{rollouts}")
        if report.values_file is not None:
            actions = report.settings.rollout_actions
            before = f" after {actions} rollout actions" if actions else ""
            parts.append(f"valued by {report.values_file}{before}")
        text = " and ".join(parts) if parts else "evaluated"
        return text[0].upper() + text[1:]

    def _table(self, header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
        return ["  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in (header, *rows)]

    def _cells(self, agreement: Agreement) -> tuple[str, str, str, str]:
        return (
            str(agreement.optimal),
            f"{agreement.optimal_visit_share:.3f}",
            f"{agreement.mean_regret:.3f}",
            f"{agreement.seconds_per_choice:.6f}",
        )
