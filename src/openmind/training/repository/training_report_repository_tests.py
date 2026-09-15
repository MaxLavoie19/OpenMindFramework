import json
from pathlib import Path

from openmind.training.mapper.training_report_json_mapper import TrainingReportJsonMapper
from openmind.training.mapper.training_report_json_mapper_tests import REPORT
from openmind.training.repository.training_report_repository import TrainingReportRepository


def test_save_writes_the_report_named_after_its_start_and_overwrites_it(tmp_path: Path) -> None:
    repository = TrainingReportRepository(TrainingReportJsonMapper())

    first = repository.save(REPORT, tmp_path)
    again = repository.save(REPORT, tmp_path)

    assert first == again == tmp_path / "chess" / "2026-09-14_13-00-00.json"
    assert json.loads(first.read_text(encoding="utf-8"))["domain"] == "chess"
