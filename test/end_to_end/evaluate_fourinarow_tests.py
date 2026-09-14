import json
from pathlib import Path

import pytest

from openmind.entrypoint.evaluate import main

pytestmark = pytest.mark.log_level("INFO")


def test_evaluate_without_positions_reports_the_baselines_only(tmp_path: Path) -> None:
    main(
        [
            "tictactoe/fourinarow",
            *("--games", "2", "--iterations", "10", "--positions", "0", "--seed", "1"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe" / "fourinarow").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["domain"] == "tictactoe/fourinarow"
    assert [(item["opponent"], item["wins"] + item["draws"] + item["losses"]) for item in report["baselines"]] == [
        ("random", 2),
        ("untrained MCTS", 2),
    ]
    assert report["agreement"] == []
    (log_file,) = (tmp_path / "log" / "tictactoe" / "fourinarow").glob("*.log")
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "INFO  openmind.evaluation.service.evaluator Agreement with perfect play skipped: no positions" in lines
    assert lines[-1] == f"INFO  openmind.entrypoint.evaluate Saved report {report_file}"


def test_a_reference_search_measures_agreement_where_exact_search_cannot_reach(tmp_path: Path) -> None:
    main(
        [
            "tictactoe/fourinarow",
            *("--games", "0", "--iterations", "5", "--positions", "2", "--budgets", "5", "--seed", "1"),
            *("--reference-iterations", "20"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe" / "fourinarow").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["settings"]["reference_iterations"] == 20
    assert [(item["iterations"], item["positions"]) for item in report["agreement"]] == [(5, 2)]
