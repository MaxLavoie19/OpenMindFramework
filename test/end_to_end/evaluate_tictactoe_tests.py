import json
from pathlib import Path

import pytest

from openmind.agent.factory.game_factory import declare_game
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.entrypoint.evaluate import main
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.model.python_rule import PythonRule
from openmind.testing.service.log_reader import said

pytestmark = pytest.mark.log_level("INFO")


def heuristics(knowledge: Path, context: str, *, move: bool = False, position: bool = False) -> str:
    """Declares a variant of tic-tac-toe carrying the heuristics asked for, as training would, and gives its name: a
    move rule rating every placement 0.5, and a position rule reading wins(me)."""
    base = create_knowledge_base("tictactoe", knowledge)
    declarer = RuleDeclarer(base, context)
    declarer.inherits(declare_game("tictactoe", base))
    if move:
        declarer.move("a placement is worth half", PythonRule("0.5 if action == 'place' else None"), 1.0)
    if position:
        declarer.position("wins(me)", PythonRule("wins(me)"), 1.0)
    return declarer.done()


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
    assert report["settings"] == {
        "games": 2,
        "iterations": 10,
        "positions": 3,
        "budgets": [5, 10],
        "seed": 1,
        "reference_iterations": None,
        "guided_rollouts": True,
        "rollout_actions": 0,
        "rollout_limit": None,
        "unfinished_payoff": None,
        "time_control": None,
        "expected_steps": 30,
        "time_reserve": 0.05,
        "selection": "ucb1",
        "puct_exploration": 1.5,
        "prior": "uniform",
        "prior_temperature": 0.1,
    }
    assert [(item["opponent"], item["wins"] + item["draws"] + item["losses"]) for item in report["baselines"]] == [
        ("random", 2),
        ("untrained MCTS", 2),
    ]
    assert [(item["iterations"], item["positions"]) for item in report["agreement"]] == [(5, 3), (10, 3)]
    assert (report["unguided_agreement"], report["rater"]) == ([], None)
    output = capsys.readouterr().out
    assert "\nAgreement with perfect play on 3 positions (every action optimal in " in output
    assert "\niterations  optimal  visits on optimal  mean regret  seconds per choice\n" in output
    assert output.endswith(f"Saved report {report_file}\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert "INFO  openmind.evaluation.service.exact_search tictactoe has 4520 positions with a legal action" in lines
    assert lines[-1] == f"INFO  openmind.entrypoint.evaluate Saved report {report_file}"


def test_evaluate_with_rules_guides_the_agent_and_records_the_rules_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rules_file = heuristics(tmp_path / "knowledge", "guided", move=True)

    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "3", "--budgets", "5", "--seed", "1"),
            *("--heuristics", rules_file, "--knowledge", str(tmp_path / "knowledge")),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["rules_file"] == rules_file
    assert report["agreement"][0]["seconds_per_choice"] >= 0.0
    assert [(item["iterations"], item["positions"]) for item in report["unguided_agreement"]] == [(5, 3)]
    assert report["rater"]["positions"] == 3
    output = capsys.readouterr().out
    assert f"\nGuided by {rules_file} against unguided, on the same 3 positions (every action optimal in " in output
    assert "\nRules alone: ratings separate actions in " in output
    assert "\nGuided against unguided, paired by position (guided minus unguided; Wilcoxon and McNemar p-values):\n" in output
    assert [test["iterations"] for test in report["guidance_tests"]] == [5]
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert f"INFO  openmind.entrypoint.evaluate Evaluating with the move rules of {rules_file}" in lines
    assert any(line.startswith("INFO  openmind.evaluation.service.evaluator Rater alone: ") for line in lines)


def test_evaluate_with_values_values_the_agent_positions_and_records_the_values_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    values_file = heuristics(tmp_path / "knowledge", "valued", position=True)

    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "3", "--budgets", "5", "--seed", "1"),
            *("--heuristics", values_file, "--knowledge", str(tmp_path / "knowledge"), "--rollout-actions", "1"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert (report["rules_file"], report["values_file"], report["rater"]) == (None, values_file, None)
    assert report["settings"]["rollout_actions"] == 1
    assert report["valuer"]["positions"] == 3
    assert [(item["iterations"], item["positions"]) for item in report["unguided_agreement"]] == [(5, 3)]
    output = capsys.readouterr().out
    assert f"\nValued by {values_file} after 1 rollout actions against unguided, on the same 3 positions " in output
    assert "\nValues alone: valued " in output
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert f"INFO  openmind.entrypoint.evaluate Evaluating with the position rules of {values_file}" in lines


def test_a_rollout_limit_applies_to_every_agent_and_is_recorded(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "3", "--budgets", "5", "--seed", "1"),
            *("--rollout-limit", "2", "--unfinished-payoff", "0.25"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    settings = json.loads(report_file.read_text(encoding="utf-8"))["settings"]
    assert (settings["rollout_limit"], settings["unfinished_payoff"]) == (2, 0.25)
    assert "\nAgreement with perfect play on 3 positions (rollout limit 2, unfinished payoff 0.25; every action optimal in " in (
        capsys.readouterr().out
    )


def test_negative_rollout_actions_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--rollout-actions", "-1", "--log-directory", str(tmp_path)])


def test_unguided_rollouts_are_recorded_and_named(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    rules_file = heuristics(tmp_path / "knowledge", "guided", move=True)

    main(
        [
            "tictactoe",
            *("--games", "0", "--iterations", "10", "--positions", "3", "--budgets", "5", "--seed", "1"),
            *("--heuristics", rules_file, "--knowledge", str(tmp_path / "knowledge"), "--rollouts", "unguided"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    assert json.loads(report_file.read_text(encoding="utf-8"))["settings"]["guided_rollouts"] is False
    output = capsys.readouterr().out
    assert f"\nGuided by {rules_file} with unguided rollouts against unguided, on the same 3 positions " in output


def test_several_workers_give_the_same_report(tmp_path: Path) -> None:
    rules_file = heuristics(tmp_path / "knowledge", "guided", move=True)
    reports = []
    for workers in ("1", "2"):
        main(
            [
                "tictactoe",
                *("--games", "4", "--iterations", "10", "--positions", "6", "--budgets", "5", "--seed", "1"),
                *("--heuristics", rules_file, "--knowledge", str(tmp_path / "knowledge"), "--workers", workers),
                *("--log-directory", str(tmp_path / workers / "log"), "--report-directory", str(tmp_path / workers / "report")),
            ]
        )
        (report_file,) = (tmp_path / workers / "report" / "tictactoe").glob("*.json")
        report = json.loads(report_file.read_text(encoding="utf-8"))
        for key in ("agreement", "unguided_agreement"):
            for item in report[key]:
                del item["seconds_per_choice"]
        del report["created_at"]
        reports.append(report)

    assert reports[0] == reports[1]


def test_all_positions_measures_every_position(tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *("--games", "0", "--positions", "all", "--budgets", "1"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["settings"]["positions"] is None
    assert [(item["iterations"], item["positions"]) for item in report["agreement"]] == [(1, 4520)]


@pytest.mark.parametrize("positions", ["-1", "some"])
def test_invalid_positions_are_rejected(positions: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--positions", positions, "--log-directory", str(tmp_path)])


def test_invalid_budgets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--budgets", "5,x", "--log-directory", str(tmp_path)])


def test_evaluate_on_a_clock_reports_the_time_control_and_the_games_won_and_lost_on_time(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "0", "--seed", "1", "--time-control", "0.05+0"),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
            *("--knowledge", str(tmp_path / "knowledge")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert (report["settings"]["time_control"], report["settings"]["expected_steps"]) == ("0.05+0", 30)
    assert [(item["time_control"], item["wins_on_time"] + item["losses_on_time"] <= 2) for item in report["baselines"]] == [
        ("0.05+0", True),
        ("0.05+0", True),
    ]
    assert "0.05+0" in capsys.readouterr().out
    assert (tmp_path / "knowledge" / "tictactoe" / "records.jsonl").is_file()


@pytest.mark.parametrize(("flag", "value"), [("--time-control", "3"), ("--time-control", "0+2"), ("--expected-steps", "0")])
def test_evaluate_refuses_a_time_control_or_expected_steps_it_can_t_read(flag: str, value: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", flag, value, "--log-directory", str(tmp_path / "log")])


@pytest.mark.parametrize(("prior", "message"), [("rater", "--prior rater needs --heuristics"), ("value", "--prior value needs --heuristics")])
def test_evaluate_refuses_a_prior_without_the_rules_it_reads(
    prior: str, message: str, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--selection", "puct", "--prior", prior, "--log-directory", str(tmp_path / "log")])

    assert message in capsys.readouterr().err


def test_evaluate_by_puct_records_the_selection_in_its_report(tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *("--games", "2", "--iterations", "10", "--positions", "0", "--seed", "1", "--selection", "puct"),
            *("--puct-exploration", "2.0", "--knowledge", str(tmp_path / "knowledge")),
            *("--log-directory", str(tmp_path / "log"), "--report-directory", str(tmp_path / "report")),
        ]
    )

    (report_file,) = (tmp_path / "report" / "tictactoe").glob("*.json")
    settings = json.loads(report_file.read_text(encoding="utf-8"))["settings"]
    assert (settings["selection"], settings["puct_exploration"], settings["prior"]) == ("puct", 2.0, "uniform")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert any(line.endswith(", puct with the uniform prior, tree depth " + line.rsplit(" ", 1)[-1]) for line in said(log_file) if "Searched " in line)
