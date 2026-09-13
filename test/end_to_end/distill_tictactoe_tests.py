from pathlib import Path

import pytest

from openmind.entrypoint.distill import main
from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository

pytestmark = pytest.mark.log_level("INFO")


def test_distill_prints_and_saves_rules(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *("--games", "2", "--held-out-games", "1", "--iterations", "20", "--min-rule-visits", "5", "--seed", "1"),
            *("--log-directory", str(tmp_path / "log"), "--rules-directory", str(tmp_path / "rules")),
        ]
    )

    (rules_file,) = (tmp_path / "rules" / "tictactoe").glob("*.json")
    rule_base = RuleBaseRepository(RuleBaseJsonMapper(ExpressionJsonMapper())).load(rules_file)
    assert rule_base.domain == "tictactoe"
    assert rule_base.rules[0].action == "place"
    output = capsys.readouterr().out
    assert output.startswith("place in any state: EV ")
    assert output.endswith(f"Saved rules {rules_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert log_file.read_text(encoding="utf-8").splitlines()[-1] == f"INFO  openmind.entrypoint.distill Saved rules {rules_file}"
