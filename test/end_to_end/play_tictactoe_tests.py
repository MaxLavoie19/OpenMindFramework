from collections.abc import Iterator
from pathlib import Path

import pytest

from openmind.entrypoint.play import main

X_WINS = ("1", "3", "1", "2", "1")  # X: (1,1) (1,2) (1,3); O: (2,1) (2,2)


def play(monkeypatch: pytest.MonkeyPatch, log_directory: Path, *answers: str, options: tuple[str, ...] = ()) -> None:
    replies: Iterator[str] = iter(answers)

    def reply(prompt: str) -> str:
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", reply)
    main(["tictactoe", *options, "--log-directory", str(log_directory)])


def log_lines(log_directory: Path) -> list[str]:
    (log_file,) = log_directory.glob("tictactoe/*.log")
    return log_file.read_text(encoding="utf-8").splitlines()


def test_game_x_wins_ends_with_the_final_payoffs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, *X_WINS)

    assert capsys.readouterr().out.endswith("payoff(O) = 0.0\npayoff(X) = 1.0\nturn = 'O'\n")


def test_invalid_input_asks_again(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, "x", "0", "10", "1")

    assert capsys.readouterr().out.count("Enter a number from 1 to 9.") == 3
    assert log_lines(tmp_path)[-1] == "INFO  openmind.entrypoint.play Input ended before the game was over"


def test_log_records_each_choice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    play(monkeypatch, tmp_path, *X_WINS)

    assert [line for line in log_lines(tmp_path) if " Chose " in line] == [
        "INFO  openmind.entrypoint.play Chose place(col=1, row=1)",
        "INFO  openmind.entrypoint.play Chose place(col=1, row=2)",
        "INFO  openmind.entrypoint.play Chose place(col=2, row=1)",
        "INFO  openmind.entrypoint.play Chose place(col=2, row=2)",
        "INFO  openmind.entrypoint.play Chose place(col=3, row=1)",
    ]


def test_agent_plays_o_until_the_game_is_over(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    play(monkeypatch, tmp_path, "1", "1", "1", "1", "1", options=("--agent", "O", "--iterations", "100", "--seed", "1"))

    assert "O chose place(" in capsys.readouterr().out
    lines = log_lines(tmp_path)
    assert any(line.startswith("INFO  openmind.mcts.service.tree_search Most visited: ") for line in lines)
    assert lines[-1] == "INFO  openmind.entrypoint.play No legal action left: game over"


def test_the_agent_can_search_with_a_rollout_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    options = ("--agent", "O", "--iterations", "9", "--seed", "1", "--rollout-limit", "0", "--unfinished-payoff", "0.25")

    play(monkeypatch, tmp_path, "1", options=options)

    assert "O chose place(" in capsys.readouterr().out
    lines = log_lines(tmp_path)
    assert "INFO  openmind.mcts.service.tree_search place(col=1, row=2): 1 visits, mean payoff 0.25 for O" in lines


def test_a_negative_rollout_limit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--rollout-limit", "-1", "--log-directory", str(tmp_path)])


def test_unknown_agent_player_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["tictactoe", "--agent", "Z", "--log-directory", str(tmp_path)])
