from datetime import datetime
from pathlib import Path

from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.mapper.value_base_json_mapper_tests import VALUE_BASE
from openmind.rbs.repository.value_base_repository import ValueBaseRepository


def test_save_then_load_gives_the_same_value_base(tmp_path: Path) -> None:
    repository = ValueBaseRepository(ValueBaseJsonMapper())

    path = repository.save(VALUE_BASE, tmp_path, datetime(2026, 9, 14, 1, 0, 0))

    assert path == tmp_path / "tictactoe" / "2026-09-14_01-00-00.json"
    assert repository.load(path) == VALUE_BASE


def test_write_puts_the_value_base_at_the_path_given(tmp_path: Path) -> None:
    repository = ValueBaseRepository(ValueBaseJsonMapper())

    path = repository.write(VALUE_BASE, tmp_path / "run" / "round-1.json")

    assert path == tmp_path / "run" / "round-1.json"
    assert repository.load(path) == VALUE_BASE
