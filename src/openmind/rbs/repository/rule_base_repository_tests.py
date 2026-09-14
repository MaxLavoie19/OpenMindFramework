from datetime import datetime
from pathlib import Path

from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository
from openmind.rule.model.python_rule import PythonRule


def test_save_then_load_gives_the_same_rule_base(tmp_path: Path) -> None:
    rule_base = RuleBase(
        "tictactoe", (Rule("place", (), 0.5, 1000), Rule("place", (PythonRule("cell[2, 2] == None"),), 0.75, 400))
    )
    repository = RuleBaseRepository(RuleBaseJsonMapper())

    path = repository.save(rule_base, tmp_path, datetime(2026, 9, 13, 17, 0, 0))

    assert path == tmp_path / "tictactoe" / "2026-09-13_17-00-00.json"
    assert repository.load(path) == rule_base
