import re
from pathlib import Path

import pytest

from openmind.entrypoint.distill_values import main
from openmind.doxastic.constant.rule_kind_constant import POSITION
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.testing.service.log_reader import said

pytestmark = pytest.mark.log_level("INFO")

SMALL = (
    *("--games", "3", "--held-out-games", "2", "--iterations", "20", "--seed", "1"),
    *("--seconds", "300", "--memory", "1", "--candidates", "1000", "--prices", "0.1,0.01", "--max-steps", "200"),
)


def test_distill_values_prints_and_declares_position_rules(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *SMALL,
            *("--workers", "1", "--log-directory", str(tmp_path / "log"), "--knowledge", str(tmp_path / "knowledge")),
        ]
    )

    declared = create_knowledge_base("tictactoe", tmp_path / "knowledge").rules("tictactoe distilled", (POSITION,))
    assert declared
    output = capsys.readouterr().out
    assert re.search(r"^[+-]\d", output)
    assert re.search(
        r"\nPosition rules: \d+ of \d+ candidate terms, chosen at price (0\.1|0\.01); declared under tictactoe distilled\n",
        output,
    )
    assert "\nprice  terms kept  steps  settled  training loss  held-out loss\n" in output
    assert re.search(
        r"\nRows: \d+ for training, \d+ held out, valued at the outcome target; mean absolute error on held-out rows: "
        r"0\.\d+\n",
        output,
    )
    assert output.endswith("Declared under tictactoe distilled\n")
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert said(log_file)[-1] == (
        "INFO  openmind.entrypoint.distill_values Declared the position rules under tictactoe distilled"
    )


def test_several_workers_distill_the_same_position_rules(tmp_path: Path) -> None:
    fitted = []
    for workers in ("1", "2"):
        main(
            [
                "tictactoe",
                *SMALL,
                *("--workers", workers, "--log-directory", str(tmp_path / workers / "log")),
                *("--knowledge", str(tmp_path / workers / "knowledge")),
            ]
        )
        rules = create_knowledge_base("tictactoe", tmp_path / workers / "knowledge").rules("tictactoe distilled", (POSITION,))
        fitted.append(tuple((rule.name, rule.weight("tictactoe distilled")) for rule in rules))

    assert fitted[0] == fitted[1]


def test_a_rollout_limit_stops_the_self_play_rollouts(tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *SMALL,
            *("--rollout-limit", "0", "--unfinished-payoff", "0.25", "--workers", "1"),
            *("--log-directory", str(tmp_path / "log"), "--knowledge", str(tmp_path / "knowledge")),
        ]
    )

    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    lines = said(log_file)
    assert "INFO  openmind.entrypoint.distill_values Self-play rollouts stop after 0 actions, every player getting 0.25" in lines
    assert any(line.startswith("INFO  openmind.mcts.service.tree_search ") and "mean payoff 0.25 for " in line for line in lines)


def test_negative_prices_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--prices", "0.1,-1", "--log-directory", str(tmp_path)])


def test_distill_values_on_a_training_clock_remembers_its_games_where_it_is_told(tmp_path: Path) -> None:
    main(
        [
            "tictactoe",
            *SMALL,
            *("--training-time-control", "0.05+0", "--knowledge", str(tmp_path / "knowledge")),
            *("--workers", "1", "--log-directory", str(tmp_path / "log")),
        ]
    )

    records = (tmp_path / "knowledge" / "tictactoe" / "records.jsonl").read_text(encoding="utf-8")
    assert records.count('\\"time_control\\": \\"0.05+0\\"') == 3 + 2
    (log_file,) = (tmp_path / "log" / "tictactoe").glob("*.log")
    assert sum(1 for line in said(log_file) if " on 0.05+0, clocks " in line) == 3 + 2
