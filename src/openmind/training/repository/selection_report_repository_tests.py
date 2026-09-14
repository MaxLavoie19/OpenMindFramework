from pathlib import Path

from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.training.mapper.selection_report_json_mapper import SelectionReportJsonMapper
from openmind.training.mapper.selection_report_json_mapper_tests import report
from openmind.training.repository.selection_report_repository import SelectionReportRepository


def test_save_writes_the_report_under_its_domain_and_start_and_overwrites_it_later(tmp_path: Path) -> None:
    mapper = SelectionReportJsonMapper(RuleTextMapper())
    repository = SelectionReportRepository(mapper)

    first = repository.save(report(False, None), tmp_path)
    second = repository.save(report(), tmp_path)

    assert first == second == tmp_path / "tictactoe" / "2026-09-13_23-45-00.json"
    assert second.read_text(encoding="utf-8") == mapper.to_json(report()) + "\n"
