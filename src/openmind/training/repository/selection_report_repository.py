from pathlib import Path

from openmind.training.mapper.selection_report_json_mapper import SelectionReportJsonMapper
from openmind.training.model.selection_report import SelectionReport


class SelectionReportRepository:
    """Saves selection reports as JSON files; saving a report again overwrites the same file."""

    def __init__(self, selection_report_json_mapper: SelectionReportJsonMapper) -> None:
        self._selection_report_json_mapper = selection_report_json_mapper

    def save(self, report: SelectionReport, directory: Path) -> Path:
        """Writes <directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json, named after when the selection started, and returns its
        path."""
        path = directory / report.domain / f"{report.created_at:%Y-%m-%d_%H-%M-%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._selection_report_json_mapper.to_json(report) + "\n", encoding="utf-8")
        return path
