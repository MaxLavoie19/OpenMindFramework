import json
from pathlib import Path

from openmind.dashboard.model.report_summary import ReportSummary
from openmind.dashboard.model.round_row import RoundRow


class ReportReader:
    """Reads the newest training report of a directory, as `TrainingReportJsonMapper` writes them, into what the page
    shows."""

    def summary(self, directory: Path) -> ReportSummary | None:
        """The newest `*.json` report in the directory, by modification time; None without one."""
        reports = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime) if directory.is_dir() else []
        if not reports:
            return None
        document = json.loads(reports[-1].read_text(encoding="utf-8"))
        rounds = tuple(self._row(item) for item in document["rounds"])
        latest = document["rounds"][-1]["value_base"]["rules"] if document["rounds"] else []
        rules = tuple(
            sorted(((rule["term"], float(rule["weight"])) for rule in latest), key=lambda rule: -abs(rule[1]))
        )
        return ReportSummary(reports[-1], document["created_at"], bool(document["complete"]), rounds, rules)

    def _row(self, item: dict[str, object]) -> RoundRow:
        chosen = item.get("chosen_price")
        fits = item.get("fits") or []
        loss = next((fit["held_out_loss"] for fit in fits if fit["price"] == chosen), None)  # type: ignore[index,union-attr]
        previous = item.get("against_previous")
        pondering = item.get("pondering")
        return RoundRow(
            int(item["number"]),  # type: ignore[call-overload]
            len(item["value_base"]["rules"]),  # type: ignore[index]
            loss,
            item.get("held_out_error"),  # type: ignore[arg-type]
            tuple((results["opponent"], self._results(results)) for results in item.get("baselines") or []),  # type: ignore[union-attr]
            None if previous is None else f"{previous['opponent']}: {self._results(previous)}",  # type: ignore[index,arg-type]
            None
            if pondering is None
            else tuple(  # type: ignore[arg-type]
                int(pondering[key])  # type: ignore[index]
                for key in ("positions", "proven", "seeds", "seeds_kept", "seeds_in_rules")
            ),
            float(item["seconds"]),  # type: ignore[arg-type]
        )

    def _results(self, results: dict[str, object]) -> str:
        return f"{results['wins']} / {results['draws']} / {results['losses']}"
