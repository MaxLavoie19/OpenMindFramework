import re
from pathlib import Path

import pytest

from openmind.entrypoint.distill import main
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
    rule_base = RuleBaseRepository(RuleBaseJsonMapper()).load(rules_file)
    assert rule_base.domain == "tictactoe"
    assert rule_base.rules[0].action == "place"
    output = capsys.readouterr().out
    assert output.startswith("place in any state: EV ")
    assert re.search(
        r"\nGoal patterns: \d+; hypotheses: \d+ tested, \d+ validated at a false discovery rate of 0\.05; \d+ covered by "
        r"a simpler rule\n",
        output,
    )
    assert output.endswith(f"Saved rules {rules_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert log_file.read_text(encoding="utf-8").splitlines()[-1] == f"INFO  openmind.entrypoint.distill Saved rules {rules_file}"


def test_several_workers_distill_the_same_rules(tmp_path: Path) -> None:
    rule_bases = []
    for workers in ("1", "2"):
        main(
            [
                "tictactoe",
                *("--games", "2", "--held-out-games", "2", "--iterations", "20", "--min-rule-visits", "5"),
                *("--seed", "1", "--workers", workers),
                *("--log-directory", str(tmp_path / workers / "log"), "--rules-directory", str(tmp_path / workers / "rules")),
            ]
        )
        (rules_file,) = (tmp_path / workers / "rules" / "tictactoe").glob("*.json")
        rule_bases.append(RuleBaseRepository(RuleBaseJsonMapper()).load(rules_file))

    assert rule_bases[0] == rule_bases[1]
    (log_file,) = (tmp_path / "2" / "log" / "tictactoe").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "INFO  openmind.entrypoint.distill Running self-play and rule generation in 2 worker processes" in lines
    assert any(line.startswith("INFO  openmind.mcts.service.tree_search Searching 20 iterations") for line in lines)
