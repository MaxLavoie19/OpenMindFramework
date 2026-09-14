import json
from datetime import datetime
from pathlib import Path

import pytest

from openmind.entrypoint.select import main
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository
from openmind.rule.model.python_rule import PythonRule

pytestmark = pytest.mark.log_level("INFO")


def test_select_saves_the_selected_rules_and_the_report(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    repository = RuleBaseRepository(RuleBaseJsonMapper())
    never_matches = Rule("place", (PythonRule("cell[1, 1] == 'Z'"),), 0.1, 10)
    candidates = RuleBase(
        "tictactoe",
        (
            Rule("place", (), 0.5, 100),
            Rule("place", (PythonRule("win_chance(action) >= 1"),), 1.0, 40, priority=True),
            never_matches,
        ),
    )
    rules_file = repository.save(candidates, tmp_path / "candidates", datetime(2026, 9, 13, 12, 0, 0))

    main(
        [
            "tictactoe",
            *("--rules", str(rules_file), "--positions", "20", "--iterations", "5", "--resamples", "200"),
            *("--workers", "2"),
            *("--log-directory", str(tmp_path / "log"), "--rules-directory", str(tmp_path / "rules")),
            *("--report-directory", str(tmp_path / "selection")),
        ]
    )

    (selected_file,) = (tmp_path / "rules" / "tictactoe").glob("*.json")
    assert never_matches not in repository.load(selected_file).rules
    (report_file,) = (tmp_path / "selection" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert (report["complete"], report["candidates_file"], report["candidates"]) == (True, str(rules_file), 3)
    assert report["confirmation"] is not None
    output = capsys.readouterr().out
    assert output.startswith("Selected ")
    assert output.endswith(f"Saved selected rules {selected_file}\nSaved selection report {report_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert lines[-1] == f"INFO  openmind.entrypoint.select Saved selection report {report_file}"
    assert any(line.startswith("INFO  openmind.training.service.rule_selector Removed place when cell[1, 1] == 'Z'") for line in lines)


def test_a_rule_base_file_is_required(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--log-directory", str(tmp_path)])
