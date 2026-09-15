from collections.abc import Sequence

from openmind.evaluation.model.match_results import MatchResults
from openmind.training.model.pondering_summary import PonderingSummary
from openmind.training.model.training_report import TrainingReport


class TrainingReportTextMapper:
    """Maps a training report to readable text: a heading, then a table with one line per round: its value rules, the
    chosen fit's held-out loss beside the held-out loss of a fit without rules, the held-out error, and its games
    against each opponent as wins / draws / losses."""

    def to_text(self, report: TrainingReport) -> str:
        state = "complete" if report.complete else "not complete"
        start = report.settings.start_file or "no value rules"
        lines = [
            f"Trained {report.domain} value rules for {len(report.rounds)} of {report.settings.rounds} rounds, {state}; "
            f"round 1 started from {start}"
        ]
        if not report.rounds:
            return "\n".join(lines)
        opponents = list(dict.fromkeys(results.opponent for item in report.rounds for results in item.baselines))
        pondered = any(item.pondering is not None for item in report.rounds)
        header = (
            "round",
            "rules",
            "held-out loss",
            "with no rule",
            "held-out error",
            *(f"against {opponent}" for opponent in opponents),
            "against the previous",
            *(("pondered: proven / seeds / kept / in rules",) if pondered else ()),
            "seconds",
        )
        rows = [
            (
                str(item.number),
                str(len(item.value_base.rules)),
                self._loss(None if item.chosen is None else item.chosen.held_out_loss),
                self._loss(next((fit.held_out_loss for fit in item.fits if fit.terms_kept == 0), None)),
                "none" if item.held_out_error is None else f"{item.held_out_error:.4f}",
                *(self._against(item.baselines, opponent) for opponent in opponents),
                "none"
                if item.against_previous is None
                else f"{item.against_previous.opponent}: {self._results(item.against_previous)}",
                *((self._pondering(item.pondering),) if pondered else ()),
                f"{item.seconds:.0f}",
            )
            for item in report.rounds
        ]
        lines.extend(self._table(header, rows))
        return "\n".join(lines)

    def _against(self, baselines: Sequence[MatchResults], opponent: str) -> str:
        """The round's games against the opponent, or none, for a round handed over before its games."""
        found = next((results for results in baselines if results.opponent == opponent), None)
        return "none" if found is None else self._results(found)

    def _pondering(self, summary: PonderingSummary | None) -> str:
        if summary is None:
            return "none"
        return f"{summary.positions}: {summary.proven} / {summary.seeds} / {summary.seeds_kept} / {summary.seeds_in_rules}"

    def _loss(self, loss: float | None) -> str:
        return "none" if loss is None else f"{loss:.6f}"

    def _results(self, results: MatchResults) -> str:
        return f"{results.wins} / {results.draws} / {results.losses}"

    def _table(self, header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
        widths = [max(len(row[column]) for row in (header, *rows)) for column in range(len(header))]
        return ["  ".join(cell.rjust(width) for cell, width in zip(row, widths, strict=True)) for row in (header, *rows)]
