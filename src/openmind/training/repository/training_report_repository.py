from pathlib import Path

from openmind.training.mapper.training_report_json_mapper import TrainingReportJsonMapper
from openmind.training.model.training_report import TrainingReport


class TrainingReportRepository:
    """Saves training reports as JSON files; saving a report again overwrites the same file."""

    def __init__(self, training_report_json_mapper: TrainingReportJsonMapper) -> None:
        self._training_report_json_mapper = training_report_json_mapper

    def save(self, report: TrainingReport, directory: Path) -> Path:
        """Writes <directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json, named after when the training started, and returns its
        path."""
        path = directory / report.domain / f"{report.created_at:%Y-%m-%d_%H-%M-%S}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._training_report_json_mapper.to_json(report) + "\n", encoding="utf-8")
        return path
