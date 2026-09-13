from pathlib import Path

from openmind.evaluation.mapper.report_json_mapper import ReportJsonMapper
from openmind.evaluation.model.evaluation_report import EvaluationReport


class ReportRepository:
    """Saves evaluation reports as JSON files."""

    def __init__(self, report_json_mapper: ReportJsonMapper) -> None:
        self._report_json_mapper = report_json_mapper

    def save(self, report: EvaluationReport, directory: Path) -> Path:
        """Writes <directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json and returns its path."""
        path = directory / report.domain / f"{report.created_at:%Y-%m-%d_%H-%M-%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._report_json_mapper.to_json(report) + "\n", encoding="utf-8")
        return path
