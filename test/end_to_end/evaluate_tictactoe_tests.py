import json
from datetime import datetime
from pathlib import Path

import pytest

from openmind.entrypoint.evaluate import main
from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.repository.rule_base_repository import RuleBaseRepository

pytestmark = pytest.mark.log_level("INFO")


def test_evaluate_prints_and_saves_a_report_and_a_log(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "3", "--budgets", "5,10", "--seed", "1"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["rules_file"] is None
    assert report["settings"] == {"games": 2, "iterations": 10, "positions": 3, "budgets": [5, 10], "seed": 1}
    assert [(item["opponent"], item["wins"] + item["draws"] + item["losses"]) for item in report["baselines"]] == [
        ("random", 2),
        ("untrained MCTS", 2),
    ]
    assert [(item["iterations"], item["positions"]) for item in report["agreement"]] == [(5, 3), (10, 3)]
    assert capsys.readouterr().out.endswith(f"Saved report {report_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "INFO  openmind.evaluation.service.exact_search tictactoe has 4520 positions with a legal action" in lines
    assert lines[-1] == f"INFO  openmind.entrypoint.evaluate Saved report {report_file}"


def test_evaluate_with_rules_guides_the_agent_and_records_the_rules_file(tmp_path: Path) -> None:
    repository = RuleBaseRepository(RuleBaseJsonMapper(ExpressionJsonMapper()))
    rules_file = repository.save(
        RuleBase("tictactoe", (Rule("place", (), 0.5, 100),)), tmp_path / "rules", datetime(2026, 9, 13, 12, 0, 0)
    )

    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "3", "--budgets", "5", "--seed", "1"),
            *("--rules", str(rules_file)),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["rules_file"] == str(rules_file)
    assert report["agreement"][0]["seconds_per_choice"] >= 0.0
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert f"INFO  openmind.entrypoint.evaluate Evaluating with rules {rules_file}" in log_file.read_text(
        encoding="utf-8"
    ).splitlines()


def test_invalid_budgets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--budgets", "5,x", "--log-directory", str(tmp_path)])
